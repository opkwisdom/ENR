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
from utils import move_to_cuda
from data_utils import load_queries, load_qrels

from config.base import load_config
from dataclasses import dataclass


args = load_config()

def load_txt_and_save_stat(args: str, path: str):
    total_stats = {str(i): 0 for i in range(0, 9)}
    count = 0
    with open(path, "r") as f:
        for line in tqdm.tqdm(f, desc="Load text file result"):
            line = line.split('\t')
            total_stats[line[3]] += 1
            count += 1
    
    for i in range(0, 9):
        total_stats[str(i)] /= count
    
    out_path = os.path.join(args.search_out_dir, "overlap_stat.json")
    with open(out_path, 'w') as f:
        json.dump(total_stats, f, indent=4, ensure_ascii=False)


def main() -> None:
    logger = _setup_logger(args.search_out_dir)
    logger.info("Args={}",format(str(args)))

    filepath = os.path.join(args.search_out_dir, "prefix_overlap.txt")
    load_txt_and_save_stat(args, filepath)


if __name__ == "__main__":
    main()
    