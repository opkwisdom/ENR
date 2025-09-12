import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup
import numpy as np

# 1. Load the model
class BaseModel(LightningModule):
    def __init__(self, config, tokenizer):
        """method used to define our model parameters"""
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            self.cfg.model_pretrained
        )
        if self.cfg.train_encoder == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # if self.cfg.model_pretrained == "roberta-base":
        self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        # self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()
        # 불러온 가중치를 torch 텐서로 변환하여  등록 (학습 불가능한 파라미터로 설정)
        self.code_book = torch.nn.Parameter(torch.load(f"../data/{self.cfg.encoder_model}/codebook_normed.pt").transpose(1, 2),requires_grad=self.cfg.train_codebook)
        self.input_soft_prompt = torch.nn.Parameter(torch.load(f"../data/{self.cfg.encoder_model}/prompt_emb.pt"))
        self.corpus = np.load(f"../data/{self.cfg.encoder_model}/smtid_ms_full.npy")

    def forward(self, input_ids, attention_mask=None):
        # RoBERTa에서 마지막 히든 스테이트 출력
        mask = attention_mask.new_ones(input_ids.shape[0], self.cfg.dep)              # [B, L]
        input_embs =  torch.cat([self.input_soft_prompt.unsqueeze(0).expand(input_ids.shape[0],-1,-1),self.model.embeddings(input_ids)], dim=1)
        attention_mask = torch.cat([mask, attention_mask], dim=1)
        outputs = self.model(inputs_embeds=input_embs, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:,:self.cfg.dep,:]
        logits = torch.einsum('bnh,nho->bno', hidden_states, self.code_book)
        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches*self.cfg.warmup_ratio,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
		"scheduler": scheduler,
		"interval": "step",
	    }

        return  [optimizer],[scheduler_config]


    def training_step(self, batch, batch_idx):
        loss,ratio= self._get_preds_loss_accuracy(batch)
        
        self.log("train_loss",loss)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        self.ranks = []
        self.hit = []
        
    def validation_step(self, batch, batch_idx):
        self._get_preds_loss_accuracy_test(batch)
    
    def on_validation_epoch_end(self):
        ranks = sum(self.ranks)/len(self.ranks)
        self.log("valid_mrr@10",ranks)
        hit = sum(self.hit)/len(self.hit)
        self.log("valid_hit@10",hit)
        
    def on_test_epoch_start(self):
        self.ranks = []
        self.hit = []
        
    def test_step(self, batch, batch_idx):
        self._get_preds_loss_accuracy_test(batch)
        
    def on_test_epoch_end(self):
        ranks = sum(self.ranks)/len(self.ranks)
        self.log("test_mrr@10",ranks)
        hit = sum(self.hit)/len(self.hit)
        self.log("test_hit@10",hit)
        
    def _get_preds_loss_accuracy(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        output = output.contiguous().view(-1, 2048) 
        labels = batch["labels"].contiguous().view(-1)
        loss = self.loss(output, labels)
        acc_ratio = (torch.argmax(output,dim=-1)==labels).sum()/labels.shape[0]
        return loss, acc_ratio
    
    def _get_preds_loss_accuracy_dev(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        tmp_label = torch.tensor([i for i in range(batch["labels"].shape[0])])        
        acc_ratio = torch.sum(torch.argmax(torch.sum(output[:,range(self.cfg.dep),batch['labels']],dim= -1),dim=-1).cpu()==tmp_label)/batch['labels'].shape[0]
        output = output.contiguous().view(-1, 2048)     # (B*L, O)
        labels = batch["labels"].contiguous().view(-1)  # (B*L,)
        loss = self.loss(output, labels)
        return loss, acc_ratio

    def _get_preds_loss_accuracy_test(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # import pdb; pdb.set_trace()
        _, indices = torch.topk(torch.sum(output[:,range(self.cfg.dep),torch.tensor(self.corpus)],dim= -1),k=10)
        for i in range(output.shape[0]):
            index = indices[i]
            positives = batch['positives'][i]
            rr = 0
            hit = 0 
            for rank, idx in enumerate(index, start=1):
                if idx in positives:
                    rr=1.0 / rank
                    hit=1
                    break
            self.ranks.append(rr)
            self.hit.append(hit)
        return
