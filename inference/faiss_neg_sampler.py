import json
import os
import glob
import tqdm
import torch
import numpy as np
import faiss
import shutil

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict
from datasets import Dataset
from typing import Dict, List, Tuple
from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    BatchEncoding
)
from pyserini.search.faiss import FaissSearcher

from logger_config import _setup_logger
from utils import move_to_cuda, save_json_to_file
from metrics import compute_mrr, trec_eval, ScoredDoc
from data_utils import load_queries, load_qrels, load_msmarco_predictions, save_preds_to_msmarco_format

from config.base import load_config
from dataclasses import dataclass


args = load_config()


@dataclass
class DenseSearchResult:
    docid: str
    score: float


@dataclass
class DocSemanticID:
    docid: str
    smtid: str


def _query_transform_func(tokenizer: PreTrainedTokenizerFast,
                        args,
                        examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples["query"],
                        max_length=args.q_max_len,
                        padding=PaddingStrategy.DO_NOT_PAD,
                        truncation=True)
    return batch_dict


@torch.no_grad()
def _worker_encode_queries(gpu_idx: int, logger) -> Tuple:
    # fail fast if shard does not exist

    query_id_to_text = load_queries(path=os.path.join(args.data_dir, args.validation_file),
                                    logger=logger)
    query_ids = sorted(list(query_id_to_text.keys()))
    queries = [query_id_to_text[query_id] for query_id in query_ids]
    dataset = Dataset.from_dict({'query_id': query_ids[:100],
                                 'query': queries[:100]})
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    # only keep data for current shard
    query_ids = dataset['query_id']
    query_id_to_text = {qid: query_id_to_text[qid] for qid in query_ids}

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.base_model)
    model: AutoModel = AutoModel.from_pretrained(args.base_model)
    model.eval()
    model.cuda()

    dataset.set_transform(partial(_query_transform_func, tokenizer, args))

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

    encoded_embeds = []
    for batch_dict in tqdm.tqdm(data_loader, desc='query encoding', mininterval=5):
        batch_dict = move_to_cuda(batch_dict)

        outputs = model(**batch_dict).last_hidden_state
        encoded_embeds.append(outputs[:,0,:].detach().cpu())    # Should be numpy array

    query_embeds = np.concatenate(encoded_embeds, axis=0)     # (N, D)
    
    logger.info('Done query encoding for worker {}'.format(gpu_idx))

    return query_embeds, query_ids, query_id_to_text


@torch.no_grad()
def _worker_batch_search(gpu_idx: int, logger, return_dict) -> Dict[str, List[DenseSearchResult]]:
    # Load faiss index
    logger.info("Load faiss dense index from {}".format(args.encode_save_dir))
    searcher: FaissSearcher = FaissSearcher(args.encode_save_dir, None)

    # Query embeddings
    query_embeds, query_ids, query_id_to_text = _worker_encode_queries(gpu_idx, logger)
    assert query_embeds.shape[0] == len(query_ids), '{} != {}'.format(query_embeds.shape[0], len(query_ids))

    query_id_to_topk = defaultdict(list)
    for i in tqdm.tqdm(range(0, len(query_ids), args.batch_size), desc="Search wtih Faiss"):
        batch_query_embed = query_embeds[i:i+args.batch_size]
        batch_query_ids = query_ids[i:i+args.batch_size]

        topk_results: Dict[str, DenseSearchResult] = searcher.batch_search(batch_query_embed, batch_query_ids, k=args.search_topk)
        for qid in topk_results.keys():
            for search_res in topk_results[qid]:
                query_id_to_topk[qid].append(
                    DenseSearchResult(
                        docid=search_res.docid,
                        score=search_res.score
                    )
                )

    return_dict[gpu_idx] = dict(query_id_to_topk)


def smtid_map(
        search_result: Dict[str, List[DenseSearchResult]],
        logger = None
    ) -> Dict[str, List[DocSemanticID]]:
    # Load semantic id file
    smtid = np.load(args.smtid_file)    # (N, L)

    query_id_to_topk_smtid = defaultdict(list)
    for qid in search_result.keys():
        for search_res in search_result[qid]:
            query_id_to_topk_smtid[qid].append(
                DocSemanticID(
                    docid=search_res.docid,
                    smtid="-".join(map(str, smtid[int(search_res.docid)]))
                )
            )

    out_path = os.path.join(args.search_out_dir, "smtid_search.txt")
    with open(out_path, "w", encoding="utf-8") as writer:
        for query_id in search_result.keys():
            for rank, doc_smtid in enumerate(query_id_to_topk_smtid[query_id]):
                writer.write('{}\t{}\t{}\t{}\n'.format(query_id, rank+1, doc_smtid.docid, doc_smtid.smtid))
    logger.info("Write docid & smtid to {} done".format(out_path))

    return query_id_to_topk_smtid


def count_overlap_prefix(
        search_smtid: Dict[str, List[DocSemanticID]],
        logger = None
    ):
    path_qrels = os.path.join(args.data_dir, '{}_qrels.txt'.format(args.search_split))
    qrels = load_qrels(path_qrels, logger)
    
    prefix_overlap_counts = defaultdict(list)

    for qid in tqdm.tqdm(search_smtid.keys(), desc="Compute overlapping prefix len"):
        pos_docids = list(qrels[qid].keys())
        if not pos_docids:
            logger.warning(f"No positive DocID found for qid={qid}")
            continue
        pos_docid = pos_docids[0]

        # positive mapping
        pos_smtid = ""
        for doc_smtid in search_smtid[qid]:
            if doc_smtid.docid == pos_docid:
                pos_smtid = doc_smtid.smtid
                break

        # count overlapping prefix between pos & neg smtid
        pos_tokens = pos_smtid.split("-")
        overlaps = []

        for doc_smtid in search_smtid[qid]:
            # if doc_smtid.docid == pos_docid:
            #     continue

            neg_tokens = doc_smtid.smtid.split("-")
            # compute overlapping prefix length
            overlap_len = 0
            intersect = 0
            for a, b in zip(pos_tokens, neg_tokens):
                if a == b:
                    overlap_len += 1
                else:
                    break
            for a, b in zip(pos_tokens, neg_tokens):
                if a == b:
                    intersect += 1
            overlaps.append((doc_smtid.smtid, overlap_len, intersect))
        
        prefix_overlap_counts[qid] = overlaps

    out_path = os.path.join(args.search_out_dir, "prefix_overlap.txt")
    with open(out_path, "w", encoding="utf-8") as writer:
        for query_id in search_smtid.keys():
            for rank, (smtid, overlap_len, intersect) in enumerate(prefix_overlap_counts[query_id]):
                writer.write('{}\t{}\t{}\t{}\t{}\n'.format(query_id, rank+1, smtid, overlap_len, intersect))
    logger.info("Write overlapping smtid length to {} done".format(out_path))



def main():
    logger = _setup_logger(args.search_out_dir)
    logger.info('Args={}'.format(str(args)))

    # Search first
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error("No gpu available")
        return
    
    logger.info("Use {} gpus".format(gpu_count))
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()

    # Test or not
    if not args.dry_run:
        torch.multiprocessing.spawn(_worker_batch_search, args=(logger, return_dict), nprocs=gpu_count)
    else:
        _worker_batch_search(0, logger, return_dict)
    
    merged = {}
    for gpu_idx, res in return_dict.items():
        merged.update(res)
    
    # DocID to smtid mapping
    search_smtid = smtid_map(merged, logger)
    count_overlap_prefix(search_smtid, logger)



if __name__ == "__main__":
    main()