import json
import os
import tqdm
import torch
import numpy as np
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
from data_utils import load_queries, load_qrels

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
    dataset = Dataset.from_dict({'query_id': query_ids,
                                 'query': queries})
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)
    if logger is not None:
        logger.info(dataset)

    # only keep data for current shard
    query_ids = dataset['query_id']
    query_id_to_text = {qid: query_id_to_text[qid] for qid in query_ids}

    if logger is not None:
        logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.base_model)
    if args.model_pretrained is not None:
        model: AutoModel = AutoModel.from_pretrained(args.model_pretrained)
    else:
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
    
    if logger is not None:
        logger.info('Done query encoding for worker {}'.format(gpu_idx))

    return query_embeds, query_ids, query_id_to_text


@torch.no_grad()
def _worker_batch_search(gpu_idx: int, logger, return_dict) -> Dict[str, List[DenseSearchResult]]:
    # Load faiss index
    if logger is not None:
        logger.info("Load faiss dense index from {}".format(args.encode_save_dir))
    searcher: FaissSearcher = FaissSearcher(args.encode_save_dir, None)

    # Query embeddings
    query_embeds, query_ids, query_id_to_text = _worker_encode_queries(gpu_idx, logger)
    assert query_embeds.shape[0] == len(query_ids), '{} != {}'.format(query_embeds.shape[0], len(query_ids))

    query_id_to_topk = defaultdict(list)
    for i in tqdm.tqdm(range(0, len(query_ids), args.batch_size), desc="Search wtih Faiss"):
        batch_query_embed = query_embeds[i:i+args.batch_size]
        batch_query_ids = query_ids[i:i+args.batch_size]

        topk_results: Dict[str, DenseSearchResult] = searcher.batch_search(batch_query_embed, batch_query_ids, k=args.search_topk + 10)
        for qid in topk_results.keys():
            for search_res in topk_results[qid]:
                query_id_to_topk[qid].append(
                    DenseSearchResult(
                        docid=str(search_res.docid),
                        score=float(search_res.score)
                    )
                )

    return_dict[gpu_idx] = {
        "results": dict(query_id_to_topk),
        "query_id_to_text": query_id_to_text
    }


def smtid_map(
        search_result: Dict[str, List[DenseSearchResult]],
        logger = None
    ) -> Dict[str, List[DocSemanticID]]:
    # Load semantic id file
    smtid = np.load(args.smtid_file)

    query_id_to_topk_smtid = defaultdict(list)
    for qid in search_result.keys():
        for search_res in search_result[qid]:
            query_id_to_topk_smtid[qid].append(
                DocSemanticID(
                    docid=search_res.docid,
                    smtid="-".join(map(str, smtid[int(search_res.docid)]))
                )
            )
    if logger is not None:
        logger.info("Mapping docid to smtid done")

    return query_id_to_topk_smtid


def save_contrastive_smtid_jsonl(
        query_id_to_topk_smtid: Dict[str, List[DocSemanticID]],
        query_texts: Dict[str, str],
        logger = None
    ) -> None:
    # Load semantic id file
    smtid = np.load(args.smtid_file)    # (N, L)
    # Load qrels file
    qrels = load_qrels(os.path.join(args.data_dir, f"{args.search_split}_qrels.txt"),
                       logger)
    out_path = os.path.join(args.search_out_dir, f"{args.search_split}_ctr_smtid.jsonl")
    with open(out_path, "w") as writer:
        for query_id in tqdm.tqdm(query_id_to_topk_smtid.keys(), desc="Saving JSONL"):
            
            positive_docids = list(qrels[query_id].keys())
            positive_smtids = ["-".join(map(str, smtid[int(pos_docid)])) for pos_docid in positive_docids]

            negative_docids = [
                doc_smtid.docid
                for doc_smtid in query_id_to_topk_smtid[query_id]
                if doc_smtid.docid not in positive_docids
            ]
            negative_smtids = [
                doc_smtid.smtid
                for doc_smtid in query_id_to_topk_smtid[query_id]
                if doc_smtid.docid not in positive_docids
            ]
            negative_docids = negative_docids[:args.search_topk]
            negative_smtids = negative_smtids[:args.search_topk]

            contents = {
                "query_id": query_id,
                "query": query_texts[query_id],
                "positives": {
                    "docid": positive_docids,
                    "smtid": positive_smtids
                },
                "negatives": {
                    "docid": negative_docids,
                    "smtid": negative_smtids
                }
            }
            writer.write(json.dumps(contents, ensure_ascii=False) + "\n")

    if logger is not None:
        logger.info("Save contrastive smtid to {}".format(out_path))


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
    
    merged_results = {}
    merged_query_texts = {}

    for gpu_idx, res in return_dict.items():
        merged_results.update(res["results"])
        merged_query_texts.update(res["query_id_to_text"])
    
    # DocID to smtid mapping
    query_id_to_topk_smtid = smtid_map(merged_results, logger)
    save_contrastive_smtid_jsonl(query_id_to_topk_smtid, merged_query_texts, logger)



if __name__ == "__main__":
    main()