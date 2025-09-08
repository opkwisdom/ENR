import os
import random
import tqdm
import json

from typing import Dict, List, Tuple, Any
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field


@dataclass
class ScoredDoc:
    qid: str
    pid: str
    rank: int
    score: float = field(default=-1)


def load_qrels(path: str, logger) -> Dict[str, Dict[str, int]]:
    assert path.endswith('.txt')

    # qid -> pid -> score
    qrels = {}
    for line in open(path, 'r', encoding='utf-8'):
        qid, _, pid, score = line.strip().split('\t')
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][pid] = int(score)

    logger.info('Load {} queries {} qrels from {}'.format(len(qrels), sum(len(v) for v in qrels.values()), path))
    return qrels


def load_queries(path: str, logger, task_type: str = 'ir') -> Tuple[Dict[str, str], Dict[str, str]]:
    qid_to_query = {}
    for line in open(path, 'r', encoding='utf-8'):
        line = json.loads(line)
        qid, query = line['query_id'], line['query']
        qid_to_query[qid] = query

    # if task_type == 'qa':
    #     qid_to_query = load_query_answers(path)
    #     qid_to_query = {k: v['query'] for k, v in qid_to_query.items()}
    # elif task_type == 'ir':
    # else:
    #     raise ValueError('Unknown task type: {}'.format(task_type))

    if logger is not None:
        logger.info('Load {} queries and positives from {}'.format(len(qid_to_query), path))
    return qid_to_query


def normalize_qa_text(text: str) -> str:
    # TriviaQA has some weird formats
    # For example: """What breakfast food gets its name from the German word for """"stirrup""""?"""
    while text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    return text


# def load_query_answers(path: str) -> Dict[str, Dict[str, Any]]:
#     assert path.endswith('.tsv')

#     qid_to_query = {}
#     for line in open(path, 'r', encoding='utf-8'):
#         query, answers = line.strip().split('\t')
#         query = normalize_qa_text(query)
#         answers = normalize_qa_text(answers)
#         qid = get_question_key(query)
#         if qid in qid_to_query:
#             logger.warning('Duplicate question: {} vs {}'.format(query, qid_to_query[qid]['query']))
#             continue

#         qid_to_query[qid] = {}
#         qid_to_query[qid]['query'] = query
#         qid_to_query[qid]['answers'] = list(eval(answers))

#     logger.info('Load {} queries from {}'.format(len(qid_to_query), path))
#     return qid_to_query


def load_corpus(path: str, logger) -> Dataset:
    assert path.endswith('.jsonl') or path.endswith('.jsonl.gz')

    # two fields: id, contents
    corpus = load_dataset('json', data_files=path)['train']
    logger.info('Load {} documents from {} with columns {}'.format(len(corpus), path, corpus.column_names))
    logger.info('A random document: {}'.format(random.choice(corpus)))
    return corpus


def load_msmarco_predictions(path: str, logger) -> Dict[str, List[ScoredDoc]]:
    assert path.endswith('.txt')

    qid_to_scored_doc = {}
    for line in tqdm.tqdm(open(path, 'r', encoding='utf-8'), desc='load prediction', mininterval=3):
        fs = line.strip().split('\t')
        qid, pid, rank = fs[:3]
        rank = int(rank)
        score = round(1 / rank, 4) if len(fs) == 3 else float(fs[3])

        if qid not in qid_to_scored_doc:
            qid_to_scored_doc[qid] = []
        scored_doc = ScoredDoc(qid=qid, pid=pid, rank=rank, score=score)
        qid_to_scored_doc[qid].append(scored_doc)

    qid_to_scored_doc = {qid: sorted(scored_docs, key=lambda sd: sd.rank)
                         for qid, scored_docs in qid_to_scored_doc.items()}

    logger.info('Load {} query predictions from {}'.format(len(qid_to_scored_doc), path))
    return qid_to_scored_doc


def save_preds_to_msmarco_format(preds: Dict[str, List[ScoredDoc]], out_path: str, logger):
    with open(out_path, 'w', encoding='utf-8') as writer:
        for qid in preds:
            for idx, scored_doc in enumerate(preds[qid]):
                writer.write('{}\t{}\t{}\t{}\n'.format(qid, scored_doc.pid, idx + 1, round(scored_doc.score, 3)))
    logger.info('Successfully saved to {}'.format(out_path))