#!/bin/bash

### Base BGE Model Eval
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src_jh/ python src_jh/inference/base_search.py \
    --base_model BAAI/bge-base-en-v1.5 \
    --do_test_only True \
    --data_dir data/msmarco \
    --search_out_dir result/search/sample \
    --validation_file dev_100.jsonl \
    --use_faiss True \
    --search_split dev \
    --search_topk 100 \
    --batch_size 64