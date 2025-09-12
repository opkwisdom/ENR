import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from torch.utils.data._utils.collate import default_collate


class Doc2DocIDDataset(Dataset):
    def __init__(self, cfg,data,tokenizer):
        self.cfg = cfg
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        max_len = self.cfg.p_max_len
        item = self.data[idx]
        input_text = item['title'].lower() + " | " + item['contents'].lower()

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



class Doc2DocIDDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        if stage == 'fit':
            with open(f"{self.cfg.data_dir}/pretrain_train.pickle", "rb") as f:
                train_data = pickle.load(f)
            with open(f"{self.cfg.data_dir}/pretrain_dev.pickle", "rb") as f:
                dev_data = pickle.load(f)

            self.train_dataset = Doc2DocIDDataset(self.cfg, train_data, self.tokenizer)
            self.valid_dataset = Doc2DocIDDataset(self.cfg, dev_data, self.tokenizer)
        elif stage == 'test':
            
            with open(f"{self.cfg.data_dir}/pretrain_dev.pickle", "rb") as file:
                dev_data = pickle.load(file)
            
            self.test_dataset = Doc2DocIDDataset(self.cfg, dev_data, self.tokenizer)

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
            self.test_dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.test_batch_size
        )