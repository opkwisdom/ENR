import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from torch.utils.data._utils.collate import default_collate


class ENR(Dataset):
    def __init__(self, cfg,data,tokenizer):
        self.cfg = cfg
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        max_len = self.cfg.p_max_len
        item = self.data[idx]
        if "query" in item.keys():
                input_text = item['query'].lower()
        else:
            input_text = item['title'].lower() + item['contents'].lower()
        inputs = self.tokenizer(
            input_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt" 
        )
        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long).squeeze(0),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long).squeeze(0),
            "labels": torch.tensor(item["smtid"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

class ENRtest(Dataset):
    def __init__(self, cfg,data,tokenizer):
        self.cfg = cfg
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        max_len = self.cfg.p_max_len
        item = self.data[idx]
        if "query" in item.keys():
                input_text = item['query'].lower()
        else:
            input_text = item['title'].lower() + item['contents'].lower()
        inputs = self.tokenizer(
            input_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt" 
        )
        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long).squeeze(0),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long).squeeze(0),
            "positives": [int(x) for x in item['positives']['doc_id']],
        }
    
    def __len__(self):
        return len(self.data)


def collate_keep_positives(batch):
    # positives만 그대로 리스트로 남기고 나머지는 기본 규칙대로
    positives = [b["positives"] for b in batch]
    rest = [{k: v for k, v in b.items() if k != "positives"} for b in batch]
    collated = default_collate(rest)
    collated["positives"] = positives   # 그대로 보존
    return collated


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        if stage == 'fit':
            if self.cfg.pretrain:
                with open(f"{self.cfg.data_dir}/{self.cfg.encoder_model}/{self.cfg.pretrain_path}", "rb") as file:
                    train_data = pickle.load(file)
            else:
                with open(f"{self.cfg.data_dir}/{self.cfg.encoder_model}/train.pickle", "rb") as file:
                    train_data = pickle.load(file)

            with open(f"{self.cfg.data_dir}/{self.cfg.encoder_model}/test.pickle", "rb") as file:
                dev_data = pickle.load(file)

            self.trn_dset = ENR(self.cfg,train_data,self.tokenizer)
            self.val_dset = ENRtest(self.cfg,dev_data,self.tokenizer)

        elif stage == 'test':
            
            with open(f"{self.cfg.data_dir}/{self.cfg.encoder_model}/test.pickle", "rb") as file:
                dev_data = pickle.load(file)
            
            self.tst_dset = ENRtest(self.cfg,dev_data,self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            batch_size=self.cfg.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.test_batch_size,
            collate_fn=collate_keep_positives
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.test_batch_size,
            collate_fn=collate_keep_positives
        )