import json
import os
import glob
import tqdm
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from functools import partial
from typing import Dict
from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    BatchEncoding
)
from logger_config import _setup_logger
from config.base import load_config
from datasets import load_dataset
from utils import move_to_cuda

args = load_config()


def _psg_transform_func(tokenizer: PreTrainedTokenizerFast,
                             args,
                             examples: Dict[str, str]) -> BatchEncoding:
    batch_dict = tokenizer(
        text = examples["title"],
        text_pair = examples["contents"],
        max_length = args.p_max_len,
        padding = PaddingStrategy.DO_NOT_PAD,
        truncation=True
    )
    return batch_dict


@torch.no_grad()
def extract_embs():
    logger = _setup_logger(args.encode_save_dir)

    def _get_out_path(layer_num: int = 0, shard_idx: int = 0) -> str:
        os.makedirs(os.path.join(args.encode_save_dir, str(layer_num)), exist_ok=True)
        return '{}/{}/shard_{}'.format(args.encode_save_dir, layer_num, shard_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.base_model)
    model: AutoModel = AutoModel.from_pretrained(args.base_model)
    model.eval()
    model.cuda()

    # Load dataset & set transform
    collection_path = os.path.join(args.data_dir, args.collection_file)
    dataset = load_dataset("json", data_files=collection_path)["train"]
    logger.info("Load {} passages from {}".format(len(dataset), collection_path))

    dataset.set_transform(partial(_psg_transform_func, tokenizer, args))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )

    num_layers = model.config.num_hidden_layers
    assert args.shard_num > 0, "shard_num should be greater than 0"
    shard_size = len(dataset) // args.shard_num
    cur_shard_num = 0
    num_encoded_docs = 0
    total_encoded_docs = 0
    logger.info("Shard size: {}".format(shard_size))

    # Encode then save emb shards
    encoded_embs = {f"layer_{i}": [] for i in range(num_layers + 1)}
    for batch_dict in tqdm(data_loader, desc="passages encoding"):
        batch_dict = move_to_cuda(batch_dict)

        outputs = model(**batch_dict, output_hidden_states=True).hidden_states  # (L+1, B, S, D)
        for layer_num in range(num_layers + 1):
            encoded_embs[f"layer_{layer_num}"].append(outputs[layer_num][:,0,:].detach().cpu())  # Only save [CLS]
        num_encoded_docs += args.batch_size

        if num_encoded_docs >= shard_size:
            for layer_num in range(num_layers + 1):
                out_path = _get_out_path(layer_num, cur_shard_num)
                concat_embeds = torch.cat(encoded_embs[f"layer_{layer_num}"], dim=0)
                logger.info('Save {} - {} embeds of {}-th layer to {}'.format(
                    total_encoded_docs, total_encoded_docs + num_encoded_docs, layer_num, out_path))
                torch.save(concat_embeds, out_path)
                encoded_embs[f"layer_{layer_num}"] = []
            
            cur_shard_num += 1
            total_encoded_docs += num_encoded_docs
            num_encoded_docs = 0
            

    if num_encoded_docs > 0:
        for layer_num in range(num_layers + 1):
            out_path = _get_out_path(layer_num, cur_shard_num)
            concat_embeds = torch.cat(encoded_embs[f"layer_{layer_num}"], dim=0)
            logger.info('Save {} - {} embeds of {}-th layer to {}'.format(
                total_encoded_docs, total_encoded_docs + num_encoded_docs, layer_num, out_path))
            torch.save(concat_embeds, out_path)
    
    logger.info("Done extract embeddings from every layers")


def extract_final_embs():
    logger = _setup_logger(args.encode_save_dir)

    def _get_out_path(shard_idx: int = 0) -> str:
        os.makedirs(os.path.join(args.encode_save_dir), exist_ok=True)
        return '{}/shard_{}'.format(args.encode_save_dir, shard_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.base_model)
    model: AutoModel = AutoModel.from_pretrained(args.model_pretrained)
    model.eval()
    model.cuda()

    # Load dataset & set transform
    collection_path = os.path.join(args.data_dir, args.collection_file)
    dataset = load_dataset("json", data_files=collection_path)["train"]
    logger.info("Load {} passages from {}".format(len(dataset), collection_path))

    dataset.set_transform(partial(_psg_transform_func, tokenizer, args))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )

    num_layers = model.config.num_hidden_layers
    assert args.shard_num > 0, "shard_num should be greater than 0"
    shard_size = len(dataset) // args.shard_num
    cur_shard_num = 0
    num_encoded_docs = 0
    total_encoded_docs = 0
    logger.info("Shard size: {}".format(shard_size))

    # Encode then save emb shards
    encoded_embs = []
    for batch_dict in tqdm(data_loader, desc="passages encoding"):
        batch_dict = move_to_cuda(batch_dict)

        outputs = model(**batch_dict).last_hidden_state  # (L+1, B, S, D)
        encoded_embs.append(outputs[:,0,:].detach().cpu())  # Only save [CLS]
        num_encoded_docs += args.batch_size

        if num_encoded_docs >= shard_size:
            out_path = _get_out_path(cur_shard_num)
            concat_embeds = torch.cat(encoded_embs, dim=0)
            logger.info('Save {} - {} embeds of to {}'.format(
                total_encoded_docs, total_encoded_docs + num_encoded_docs, out_path))
            torch.save(concat_embeds, out_path)
            encoded_embs = []
            
            cur_shard_num += 1
            total_encoded_docs += num_encoded_docs
            num_encoded_docs = 0

    if num_encoded_docs > 0:
        out_path = _get_out_path(cur_shard_num)
        concat_embeds = torch.cat(encoded_embs, dim=0)
        logger.info('Save {} - {} embeds of to {}'.format(
            total_encoded_docs, total_encoded_docs + num_encoded_docs, out_path))
        torch.save(concat_embeds, out_path)
    
    logger.info("Done extract embeddings from final layer")


if __name__ == "__main__":
    # extract_embs()
    extract_final_embs()