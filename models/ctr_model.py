import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup
import numpy as np


class CTRModel(LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            self.cfg.model_pretrained
        )
        if not self.cfg.train_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.loss = torch.nn.CrossEntropyLoss()

        codebook = torch.load(f"{self.cfg.encoder_model_dir}/codebook_normed.py").transpose(1,2)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=self.cfg.train_codebook)
        
