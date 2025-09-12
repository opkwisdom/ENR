import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, BertModel, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
import einops
import copy
from dataclasses import dataclass
from typing import List


@dataclass
class Beam:
    prefix: List[int]
    logprob: float



class Doc2DocidLayerWiseModel(LightningModule):
    """
    Train D -> DocID
    SMTID are learned using a soft-prompt (Cross-Entropy)
    SMTID are determined in layer-wise
    Default layer: [2, 4, 6, 8, 9, 10, 11, 12]
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            self.cfg.model_pretrained
        )
        # freeze base encoder
        if not self.cfg.train_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        # Codebook was made using bge-base-en-v1.5 embeddings (fixed)
        codebook = torch.load(f"{self.cfg.codebook_dir}/codebook_normed.pt", weights_only=True).transpose(1,2)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=False)   # [L, D, K]
        self.soft_prompt = torch.nn.Parameter(torch.zeros(self.cfg.dep, self.model.config.hidden_size))  # [dep, D]
        torch.nn.init.xavier_normal_(self.soft_prompt.data)
        self.smtid = np.load(f"{self.cfg.codebook_dir}/smtid_ms_full.npy")

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        device = input_ids.device

        word_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L, D]
        soft_prompts = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)    # [B, dep, D]
        inputs_embeds = torch.cat([
            word_embeds[:, :1, :],  # [CLS]
            soft_prompts,           # soft prompt
            word_embeds[:, 1:, :]   # document tokens
        ], dim=1)

        sp_mask = torch.ones(B, self.cfg.dep, dtype=attention_mask.dtype, device=device)
        new_attention_mask = torch.cat([attention_mask[:, :1], sp_mask, attention_mask[:, 1:]], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            encoder_attention_mask=new_attention_mask,
            output_hidden_states=True
        )
        # Layer-wise smtid selection
        h_all = outputs.hidden_states
        layer_ids = self.cfg.smtid_layer
        logits_list = []
        for j, lid in enumerate(layer_ids):
            h_layer = h_all[lid]
            h_j = h_layer[:, 1+j, :]    # j-th soft prompt, [B, D]
            logits_j = einops.einsum(h_j, self.codebook[j],
                                     "B D, D K -> B K")            # [D, K]
            logits_list.append(logits_j)
        logits = torch.stack(logits_list, dim=1)

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
        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        mean_loss, slot_loss = self._get_preds_loss_accuracy(batch)
        self.log("train/loss", mean_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for i, sl in enumerate(slot_loss):
            self.log(f"train/loss_slot{i+1}", sl, on_step=True, on_epoch=True, sync_dist=True)

        return {"loss": mean_loss}

    def on_validation_epoch_start(self):
        self.dev_prefix_accs = {f"Acc{k}@p{i}": [] for i in range(1, self.cfg.dep+1) for k in self.cfg.prefix_accuracy}
        self.dev_losses = []
        self.dev_slot_losses = {f"slot{i}": [] for i in range(1, self.cfg.dep+1)}

    def validation_step(self, batch, batch_idx):
        mean_loss, slot_loss, acc_dict = self._get_preds_loss_accuracy_dev(batch)
        
        self.dev_losses.append(mean_loss.item())
        for i in range(len(slot_loss)):
            self.dev_slot_losses[f"slot{i+1}"].append(slot_loss[i].item())

        for key, val in acc_dict.items():
            self.dev_prefix_accs[key].append(val)
    
    def on_validation_epoch_end(self):
        for key, vals in self.dev_prefix_accs.items():
            score = sum(vals)/len(vals) if len(vals) > 0 else 0.0
            self.log(f"valid/{key}", score, prog_bar=False, on_epoch=True, sync_dist=True)

        if len(self.dev_losses) > 0:
            avg_loss = sum(self.dev_losses) / len(self.dev_losses)
            self.log("valid/loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        for key in self.dev_slot_losses.keys():
            avg_loss = sum(self.dev_slot_losses[key]) / len(self.dev_slot_losses[key])
            self.log(f"valid/{key}", avg_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        
    def on_test_epoch_start(self):
        self.test_prefix_accs = {f"Acc{k}@p{i}": [] for i in range(1, self.cfg.dep+1) for k in self.cfg.prefix_accuracy}
        
    def test_step(self, batch, batch_idx):
        acc_dict = self._get_preds_loss_accuracy_test(batch)
        for key, val in acc_result.items():
            self.test_prefix_accs[key].append(val)
        
    def on_test_epoch_end(self):
        for key, vals in self.test_prefix_accs.items():
            score = sum(vals)/len(vals) if len(vals) > 0 else 0.0
            self.log(f"test/{key}", score, prog_bar=False, on_epoch=True, sync_dist=True)
        
    def _get_preds_loss_accuracy(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) # [B, L, K]
        B, L, K = output.shape
        output = output.contiguous().view(-1, self.cfg.nbit)    # [B*L, K]
        labels = batch["labels"].contiguous().view(-1)  # [B*L,]
        
        loss = self.loss(output, labels)
        loss_mat = loss.view(B, L)
        slot_loss = loss_mat.mean(dim=0)
        slot_loss = [sl.item() for sl in slot_loss]
        mean_loss = loss.mean()

        return mean_loss, slot_loss
    
    def _get_preds_loss_accuracy_dev(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = batch["labels"]
        acc_dict = self.beam_search_prefix_acc(output, labels, acc_levels=self.cfg.prefix_accuracy)
        
        B, L, K = output.shape
        output = output.contiguous().view(-1, self.cfg.nbit)     # [B*L, K]
        labels = batch["labels"].contiguous().view(-1)  # [B*L,]

        # smt token-wise loss
        loss_vec = self.loss(output, labels)
        loss_mat = loss_vec.view(B, L)
        slot_loss = loss_mat.mean(dim=0)    # [B]
        mean_loss = slot_loss.mean()

        return mean_loss, slot_loss, acc_dict

    def _get_preds_loss_accuracy_test(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) # [B, L, K]
        labels = batch["labels"]    # [B, L]
        acc_dict = self.beam_search_prefix_acc(output, labels, acc_levels=self.cfg.prefix_accuracy)
        return acc_dict
    
    def beam_search_prefix_acc(self, output, labels, beam_size=5, acc_levels=[1,5,10,20]):
        """
        output: [B, L, K]
        labels: [B, L]
        """
        B, L, K = output.shape
        acc_dict = {f"Acc{k}@p{i}": [] for i in range(1, L+1) for k in acc_levels}

        for b in range(B):
            beam_list: List[Beam] = [Beam([], 0.0)]

            for l in range(L):
                log_probs = output[b, l].log_softmax(-1)    # [K]
                new_beam = []
                for candidate in beam_list:
                    topk = torch.topk(log_probs, beam_size)
                    for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                        new_prefix = candidate.prefix + [idx]
                        new_logprob = candidate.logprob + val
                        new_beam.append(Beam(new_prefix, new_logprob))
                # Preserve Top-k beams
                new_beam = sorted(new_beam, key=lambda x: x.logprob, reverse=True)[:beam_size]
                beam_list = new_beam

                # prefix accuracy
                gold_prefix = labels[b, :l+1].tolist()
                for k in acc_levels:
                    topk_seqs = [cand.prefix for cand in beam_list[:k]]
                    hit = any(seq[:l+1] == gold_prefix for seq in topk_seqs)
                    acc_dict[f"Acc{k}@p{l+1}"].append(float(hit))

        acc_result = {k: sum(v)/len(v) if len(v) > 0 else 0.0 for k,v in acc_dict.items()}
        return acc_result



class Doc2DocidSDModel(LightningModule):
    """
    Train D -> DocID
    There are two types of loss
    - SMTID are learned using a soft-prompt (Cross-Entropy)
    - Shallow Decoder is learned using Auto-Encoder scheme
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            self.cfg.model_pretrained
        )
        # shallow decoder model for Autoencoder training
        self.shallow_decoder = copy.deepcopy(self.model)
        self.shallow_decoder.encoder.layer = self.model.encoder.layer[-2:]    # last two layers

        if not self.cfg.train_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.resize_token_embeddings(len(self.tokenizer.vocab))
        
        self.loss = torch.nn.CrossEntropyLoss(reduction=None)
        # Codebook was made using bge-base-en-v1.5 embeddings (fixed)
        codebook = torch.load(f"{self.cfg.encoder_model_dir}/codebook_normed.pt", weights_only=True).transpose(1,2)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=False)   # [B, D, K]
        self.soft_prompt = torch.nn.Parameter(torch.zeros(cfg.dep, self.model.config.hidden_size))  # [L, D]
        torch.nn.init.xavier_normal_(self.soft_prompt.data)
        self.smtid = np.load(f"{self.cfg.encoder_model_dir}/smtid_ms_full.npy")

    def forward(self, input_ids, attention_mask=None):
        B, L = attention_mask.shape
        mask = attention_mask.new_ones(input_ids.shape[0], self.cfg.dep)
        outputs = self.model(
            input_ids=input_ids,
            encoder_attention_mask=mask
        )
        hidden_states = outputs.last_hidden_state[:, :self.cfg.dep, :]  # [B, L, D]
        logits = torch.einsum("BLD,LDK->BLK", hidden_states, self.codebook)
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
        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        loss, ratio= self._get_preds_loss_accuracy(batch)
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
        output = output.contiguous().view(-1, self.cfg.nbit) 
        labels = batch["labels"].contiguous().view(-1)
        loss = self.loss(output, labels)
        acc_ratio = (torch.argmax(output,dim=-1)==labels).sum()/labels.shape[0]
        return loss, acc_ratio
    
    def _get_preds_loss_accuracy_dev(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        tmp_label = torch.tensor([i for i in range(batch["labels"].shape[0])])        
        acc_ratio = torch.sum(torch.argmax(torch.sum(output[:,range(self.cfg.dep),batch['labels']],dim= -1),dim=-1).cpu()==tmp_label)/batch['labels'].shape[0]
        output = output.contiguous().view(-1, self.cfg.nbit)     # (B*L, O)
        labels = batch["labels"].contiguous().view(-1)  # (B*L,)
        loss = self.loss(output, labels)
        return loss, acc_ratio

    def _get_preds_loss_accuracy_test(self, batch):
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # import pdb; pdb.set_trace()
        _, indices = torch.topk(torch.sum(output[:,range(self.cfg.dep),torch.tensor(self.smtid)],dim= -1),k=10)
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