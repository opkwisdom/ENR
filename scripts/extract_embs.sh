#!/bin/bash

### Base BGE Model Extraction
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python src/inference/extract_embs.py \
#     --base_model BAAI/bge-base-en-v1.5 \
#     --data_dir data/msmarco \
#     --collection_file passages.jsonl.gz \
#     --encode_save_dir /hdd/work/ENR/data/layer_emb \
#     --p_max_len 144 \
#     --batch_size 256 \
#     --shard_num 20


### BGE Model Extraction from D -> DocID Pretrained Model
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python src/inference/extract_embs.py \
    --base_model bert-base-uncased \
    --model_pretrained pretrained_model/bge_pretrained \
    --data_dir data/msmarco \
    --collection_file passages.jsonl.gz \
    --encode_save_dir /hdd/work/ENR/data/final_emb/doc2docid \
    --p_max_len 144 \
    --batch_size 256 \
    --shard_num 20