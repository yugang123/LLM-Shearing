import os
import sys
import warnings
from types import MethodType
from typing import Any, Dict

import torch
from composer import Logger, State, Trainer
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.core import Evaluator, Event
from composer.loggers import FileLogger
from composer.optim import DecoupledAdamW
from composer.utils import dist, get_device, reproducibility
from llmfoundry.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                              DecoupledLionW, DecoupledLionW_8bit)
from llmfoundry.utils.builders import (build_algorithm, build_callback,
                                       build_logger, build_scheduler)
from llmfoundry.utils.config_utils import (log_config, pop_config,
                                           update_batch_size_info)
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch import nn
from torch.optim.optimizer import Optimizer

from llmshearing.callbacks.callbacks import DebugCallback
from llmshearing.callbacks.dynamic_loading_callback import DynamicLoadingCallback
from llmshearing.callbacks.pruning_callback import PruningCallback
from llmshearing.datasets.load_text_dataloader import build_text_dataloader
from llmshearing.models.model_registry import COMPOSER_MODEL_REGISTRY

# Clean stale shared memory for fresh execution
streaming.base.util.clean_stale_shared_memory()

def build_code_generation_model(cfg: DictConfig):
    """Initialize model for code generation tasks."""
    warnings.filterwarnings(action='ignore', message='Torchmetrics.*')
    return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)

def build_code_generation_evaluator(cfg: DictConfig, model):
    """Setup evaluation for code generation tasks with appropriate metrics."""
    eval_loader = Evaluator(
        label='eval',
        dataloader=build_text_dataloader(
            cfg.eval_loader, cfg.device_eval_batch_size, dynamic=False,
            set_names=cfg.callbacks.data_loading.set_names),
        metric_names=['BLEU', 'CodeBLEU', 'accuracy'])  # Updated for code generation
    return [eval_loader]

def load_model_weights(cfg: DictConfig):
    """Load weights for fine-tuning model initialization."""
    if cfg.model.get('path'):
        state_dict = torch.load(cfg.model.path)
        if "state" in state_dict:
            state_dict = state_dict["state"]["model"]
        print("Loaded model weights from path:", cfg.model.path)
        return state_dict
    return None

def setup_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any]) -> Optimizer:
    """Configure optimizer with appropriate parameter groups for SFT."""
    param_groups = [{"params": [p for n, p in model.named_parameters() if "l0_module" not in n],
                     "lr": optimizer_cfg.lr}]
    if 'lag_lr' in optimizer_cfg:
        lag_lr = pop_config(optimizer_cfg, "lag_lr")
        l0_params = [p for n, p in model.named_parameters() if "l0_module" in n]
        param_groups.extend([{"params": l0_params, "lr": lag_lr}])

    if optimizer_cfg.pop("name") == 'decoupled_adamw':
        return DecoupledAdamW(param_groups, **optimizer_cfg)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_cfg["name"]}')

def main(cfg):
    """Main function to initialize SFT training for code generation."""
    print("Starting SFT training for code generation...")
    warnings.filterwarnings(action='ignore', category=UserWarning)

    cfg.dist_timeout = cfg.get('dist_timeout', 1800.0)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

    reproducibility.seed_all(cfg.seed)
    cfg.run_name = cfg.get('run_name', os.environ.get('COMPOSER_RUN_NAME', 'codegen'))
    cfg = update_batch_size_info(cfg)
    fsdp_config = om.to_container(cfg.get('fsdp_config'), resolve=True) if cfg.get('fsdp_config') else None
    init_device = cfg.model.get('init_device', 'gpu')

    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get('loggers', {}).items()]
    save_folder = cfg.save_folder.replace('{run_name}', cfg.run_name)
    filename = f"{save_folder}/logs.txt"
    loggers.append(FileLogger(filename=filename, buffer_size=1, flush_interval=50))

    # Initialize the model
    model = build_code_generation_model(cfg.model)
    state_dict = load_model_weights(cfg)
    if state_dict:
        model.load_state_dict(state_dict, strict=False)

    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'Model Parameters: {cfg.n_params}')

    # Train and Evaluation Loaders
    train_loader = build_text_dataloader(cfg.train_loader, cfg.device_train_batch_size, 
                                         cfg.callbacks.data_loading.dynamic, 
                                         cfg.callbacks.data_loading.set_names)
    evaluators = build_code_generation_evaluator(cfg, model)

    # Optimizer and Scheduler
    optimizer = setup_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(cfg.scheduler.pop("name"), cfg.scheduler)

    # Callbacks for SFT
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get('callbacks', {}).items()]
    if model.model.l0_module is not None:
        callbacks.append(PruningCallback(save_folder=cfg.save_folder))

    # Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        loggers=loggers,
        callbacks=callbacks,
        device_train_microbatch_size=cfg.get('device_train_microbatch_size', 'auto'),
        fsdp_config=fsdp_config,
        save_folder=cfg.get('save_folder'),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path'),
        load_weights_only=cfg.get('load_weights_only', False),
        dist_timeout=cfg.dist_timeout,
        autoresume=cfg.autoresume,
    )

    if cfg.get('eval_first', False):
        trainer.eval()

    print('Starting training...')
    trainer.fit()
    print('Training complete.')

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    save_dir = cfg.save_folder.replace("{run_name}", cfg.run_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(cfg, save_dir + "/config.pt")

    main(cfg)
