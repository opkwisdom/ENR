import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from tqdm import tqdm
import json
import os
from torch.utils.data._utils.collate import default_collate

from dataclasses import dataclass
from typing import List


@dataclass
class DocSemanticID:
    """
    docid: "7264308"
    smtid: "229-11-340-1070-1090-1972-1464-623"
    """
    docid: str
    smtid: str

    def convert_smtid_to_list(self) -> List[int]:
        return list(map(int, self.smtid.split("-")))
    

@dataclass
class CTRExample:
    query_id: str
    query: str
    positives: List[DocSemanticID]
    negatives: List[DocSemanticID]

    def get_positive_labels(self) -> List[List[int]]:
        return [p.convert_smtid_to_list() for p in self.positives]

    def get_negative_labels(self) -> List[List[int]]:
        return [n.convert_smtid_to_list() for n in self.negatives]


class CTRDataset(Dataset):
    def __init__(self, cfg, data: List[CTRExample], tokenizer):
        self.cfg = cfg
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        max_len = self.cfg.q_max_len
        example = self.data[idx]

        query_inputs = self.tokenizer(
            example.query,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        pos_labels = example.get_negative_labels()
        neg_labels = example.get_negative_labels()

        return {
            "query_id": example.query_id,
            "input_ids": query_inputs["input_ids"].squeeze(0),
            "attention_mask": query_inputs["attention_mask"].squeeze(0),
            "positives": pos_labels,
            "negatives": neg_labels
        }

    def __len__(self):
        return len(self.data)
    

class CTRTestDataset(Dataset):
    def __init__(self, cfg, data: List[CTRExample], tokenizer):
        self.cfg = cfg
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        max_len = self.cfg.q_max_len
        example = self.data[idx]

        query_inputs = self.tokenizer(
            example.query,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        pos_labels = example.get_negative_labels()
        neg_labels = example.get_negative_labels()

        return {
            "query_id": example.query_id,
            "input_ids": query_inputs["input_ids"].squeeze(0),
            "attention_mask": query_inputs["attention_mask"].squeeze(0),
            "positives": pos_labels,
            "negatives": neg_labels
        }
    
    def __len__(self):
        return len(self.data)


class CTRDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.valid_dataset = None

    def _preprocess_data(self, example) -> CTRExample:
        _data = json.loads(example)
        _positives: List[DocSemanticID] = []
        for docid, smtid in zip(_data["positives"]):
            _positives.append(
                DocSemanticID(
                    docid=docid,
                    smtid=smtid
                )
            )
        _negatives: List[DocSemanticID] = []
        for docid, smtid in zip(_data["negatives"]):
            _negatives.append(
                docid=docid,
                smtid=smtid
            )
        
        data = CTRExample(
            query_id=_data["query_id"],
            query=_data["query"],
            positives=_positives,
            negatives=_negatives
        )

        return data


    def setup(self, stage: str):
        if stage == "fit":
            train_data: List[CTRExample] = []
            valid_data: List[CTRExample] = []

            with open(f"{self.cfg.data_dir}/train_ctr_smtid.jsonl") as file:
                for line in tqdm(file, desc="Load train file"):
                    data = self._preprocess_data(line)
                    train_data.append(data)
            with open(f"{self.cfg.data_dir}/dev_ctr_smtid.jsonl") as file:
                for line in tqdm(file, desc="Load valid file"):
                    data = self._preprocess_data(line)
                    valid_data.append(data)

            self.train_dataset = CTRDataset(self.cfg, train_data, self.tokenizer)
            self.valid_dataset = CTRTestDataset(self.cfg, valid_data, self.tokenizer)

        elif stage == "test":
            valid_data: List[CTRExample] = []

            with open(f"{self.cfg.data_dir}/dev_ctr_smtid.jsonl") as file:
                for line in tqdm(file, desc="Load valid file"):
                    data = self._preprocess_data(line)
                    valid_data.append(data)

            self.valid_dataset = CTRTestDataset(self.cfg, valid_data, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            batch_size=self.cfg.batch_size
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.test_batch_size
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.test_batch_size
        )