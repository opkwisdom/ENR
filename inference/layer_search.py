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

def load_dense_index_from_faiss(index_dir: str = None):
    """
    Load Dense index to initiate FaissSearcher
    index_dir must contain two data
    - index: faiss dense index file
    - docid: docid list file
    """
    if index_dir is not None:
        searcher = FaissSearcher(index_dir, None)
        return searcher
    elif "bge" in args.base_model:
        searcher = FaissSearcher.from_prebuilt_index('msmarco-v1-passage.bge-base-en-v1.5', None)
        return searcher
    else:
        raise NotImplementedError


def load_doc_embeds(embeds_path_list: List[str]):
    doc_embeds = []
    for _path in tqdm.tqdm(embeds_path_list, "Load documents"):
        doc_embeds.append(torch.load(_path, weights_only=True))
    embeds = torch.cat(doc_embeds, dim=0)
    embeds = embeds.numpy()
    return embeds


def write_index(index_dir, doc_embeds):
    dim = doc_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeds)

    # write index and docid
    faiss.write_index(index, os.path.join(index_dir, "index"))
    collection_id_path = os.path.join(args.data_dir, args.collection_id_file)
    shutil.copy(collection_id_path, os.path.join(index_dir, "docid"))


def _get_all_shards_path(logger) -> List[str]:
    if args.layer_num >= 0:
        path_list = glob.glob('{}/{}/shard_*'.format(args.encode_save_dir, args.layer_num))
    else:
        path_list = glob.glob('{}/shard_*'.format(args.encode_save_dir))
    assert len(path_list) > 0

    def _parse_shard_idx(p: str) -> Tuple:
        return int(os.path.basename(p).split("_")[-1])

    path_list = sorted(path_list, key=lambda path: _parse_shard_idx(path))
    logger.info('Embeddings path list: {}'.format(path_list))
    return path_list


def _get_topk_result_save_path(worker_idx: int) -> str:
    os.makedirs(args.search_out_dir, exist_ok=True)
    return '{}/{}/{}/top{}_{}.txt'.format(args.search_out_dir, str(args.search_topk), str(args.layer_num), args.search_topk, worker_idx)


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
def _worker_batch_search(gpu_idx: int, logger) -> Tuple:
    embeds_path_list = _get_all_shards_path(logger)

    # Load document embeds & write faiss index
    logger.info("Load document embeds")
    doc_embeds = load_doc_embeds(embeds_path_list)
    logger.info("Document embeds: ({}, {})".format(doc_embeds.shape[0], doc_embeds.shape[1]))
    index_dir = os.path.join(args.encode_save_dir, str(args.layer_num), "faiss_index")
    os.makedirs(index_dir, exist_ok=True)
    logger.info("Write Faiss index to {}".format(index_dir))
    write_index(index_dir, doc_embeds)

    # Load faiss index
    searcher: FaissSearcher = load_dense_index_from_faiss(index_dir)

    query_embeds, query_ids, query_id_to_text = _worker_encode_queries(gpu_idx, logger)
    assert query_embeds.shape[0] == len(query_ids), '{} != {}'.format(query_embeds.shape[0], len(query_ids))
    
    query_id_to_topk = defaultdict(list)
    for i in tqdm.tqdm(range(0, len(query_ids), args.batch_size), desc="Search wtih Faiss"):
        batch_query_embed = query_embeds[i:i+args.batch_size]
        batch_query_ids = query_ids[i:i+args.batch_size]

        topk_results: Dict[str, DenseSearchResult] = searcher.batch_search(batch_query_embed, batch_query_ids, k=args.search_topk)
        for qid in topk_results.keys():
            for search_res in topk_results[qid]:
                query_id_to_topk[qid].append((search_res.score, search_res.docid))
    
    out_path = _get_topk_result_save_path(worker_idx=gpu_idx)
    with open(out_path, 'w', encoding="utf-8") as writer:
        for query_id in query_id_to_text:
            for rank, (score, doc_id) in enumerate(query_id_to_topk[query_id]):
                writer.write('{}\t{}\t{}\t{}\n'.format(query_id, doc_id, rank + 1, round(score, 4)))
    logger.info("Write scores to {} done".format(out_path))


def _compute_and_save_metrics(worker_cnt: int, logger):
    preds: Dict[str, List[ScoredDoc]] = {}

    for worker_idx in range(worker_cnt):
        path = _get_topk_result_save_path(worker_idx)
        preds.update(load_msmarco_predictions(path, logger))

    out_path = os.path.join(args.search_out_dir, str(args.search_topk), str(args.layer_num), '{}.msmarco.txt'.format(args.search_split))
    save_preds_to_msmarco_format(preds, out_path, logger)

    logger.info('Merge done: save {} predictions to {}'.format(len(preds), out_path))

    path_qrels = os.path.join(args.data_dir, '{}_qrels.txt'.format(args.search_split))
    if os.path.exists(path_qrels):
        qrels = load_qrels(path=path_qrels, logger=logger)
        all_metrics = trec_eval(qrels=qrels, predictions=preds)

        pools = [10, 20, 50, 100]
        for k in pools:
            if k <= args.search_topk:
                all_metrics[f'mrr@{k}'] = compute_mrr(
                    qrels=qrels,
                    predictions=preds,
                    k=k
                )

        logger.info('{} trec metrics = {}'.format(args.search_split, json.dumps(all_metrics, ensure_ascii=False, indent=4)))
        save_json_to_file(all_metrics, os.path.join(args.search_out_dir, str(args.search_topk), str(args.layer_num), 'metrics_{}.json'.format(args.search_split)))
    else:
        logger.warning('No qrels found for {}'.format(args.search_split))

    # do some cleanup
    for worker_idx in range(worker_cnt):
        path = _get_topk_result_save_path(worker_idx)
        os.remove(path)


@torch.no_grad()
def write_faiss_index(logger) -> Tuple:
    embeds_path_list = _get_all_shards_path(logger)

    # Load document embeds & write faiss index
    logger.info("Load document embeds")
    doc_embeds = load_doc_embeds(embeds_path_list)
    logger.info("Document embeds: ({}, {})".format(doc_embeds.shape[0], doc_embeds.shape[1]))
    index_dir = os.path.join(args.encode_save_dir, "faiss_index")
    os.makedirs(index_dir, exist_ok=True)
    logger.info("Write Faiss index to {}".format(index_dir))
    write_index(index_dir, doc_embeds)



def _batch_search_queries():
    search_out_dir = os.path.join(args.search_out_dir, str(args.search_topk), str(args.layer_num))
    logger = _setup_logger(search_out_dir)
    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    # Test or not
    if not args.dry_run:
        torch.multiprocessing.spawn(_worker_batch_search, args=(logger,), nprocs=gpu_count)
    else:
        _worker_batch_search(0, logger)
    logger.info('Done batch search queries')

    _compute_and_save_metrics(gpu_count, logger)
    

if __name__ == "__main__":
    _batch_search_queries()
    # logger = _setup_logger(args.search_out_dir)
    # logger.info('Args={}'.format(str(args)))
    # write_faiss_index(logger)