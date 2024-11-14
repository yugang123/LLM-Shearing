""" Load text dataloader specifically for supervised fine-tuning on question-code pairs. """
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import transformers
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from llmshearing.datasets.streaming_dataset import (
    TextDynamicStreamingDataset, TextStreamingDataset)


def build_text_dataloader(cfg: DictConfig, device_batch_size: int, dynamic: bool = False, 
                          set_names: str = None, proportion: List[float] = None) -> DataLoader:
    """Builds a text dataloader for SFT on question-code pairs.

    Args:
        cfg (DictConfig): Configuration dictionary.
        device_batch_size (int): Batch size for one single device.
        dynamic (bool, optional): Whether to use dynamic streaming dataset to load data from each 
        domain dynamically. Defaults to False.
        set_names (str, optional): Name of the dataset. Defaults to None.
        proportion (List[float], optional): Initial proportion of each domain in the dataset. Defaults to None.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    
    if dynamic:
        dataset = TextDynamicStreamingDataset(local=cfg.dataset.local,
                                              max_seq_len=cfg.dataset.max_seq_len,
                                              batch_size=device_batch_size,
                                              shuffle=cfg.dataset.get('shuffle', False),
                                              shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
                                              num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
                                              proportion=proportion,
                                              set_names=set_names,
                                              is_uint16=cfg.dataset.get("is_uint16", False))
    else:
        dataset = TextStreamingDataset(
            local=cfg.dataset.local,
            max_seq_len=cfg.dataset.max_seq_len,
            split=cfg.dataset.get('split', None),
            shuffle=cfg.dataset.get('shuffle', False),
            shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
            num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
            batch_size=device_batch_size,
            is_uint16=cfg.dataset.get("is_uint16", False))

    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_name)
    COLLATE_FN = DataCollatorForQuestionCodeSFT
    collate_fn = COLLATE_FN(
        tokenizer=tokenizer,
        set_names=set_names,
        pad_to_multiple_of=cfg.dataset.get('pad_to_multiple_of', 8),
        mlm=False
    )
    
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )


@dataclass
class DataCollatorForQuestionCodeSFT:
    """ Data collator used for supervised fine-tuning on question-code pairs. """
    tokenizer: PreTrainedTokenizerBase
    set_names: List[str]
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    mlm: bool = False

    def __post_init__(self):
        self.set_name_to_id = defaultdict(int)
        self.set_name_to_id.update({name: i for i, name in enumerate(self.set_names)})

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        """Processes question-code pairs for model input and labels."""
        input_texts = [f"{feature['question']} [SEP] {feature['code']}" for feature in features]
        encodings = self.tokenizer(
            input_texts,
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        labels = encodings["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        batch = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
            "set": torch.tensor([self.set_name_to_id[feature["set"]] for feature in features])
        }
        
        if "idx" in features[0]:
            batch["idx"] = torch.tensor([feature["idx"] for feature in features])

        return batch
