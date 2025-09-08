#!/bin/bash

### Sample
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python src/inference/layer_search.py \
#     --base_model BAAI/bge-base-en-v1.5 \
#     --do_test_only True \
#     --data_dir data/msmarco \
#     --search_out_dir result/search/sample \
#     --validation_file dev_100.jsonl \
#     --use_faiss True \
#     --search_split dev \
#     --q_max_len 32 \
#     --search_topk 100 \
#     --batch_size 64 \
#     --encode_save_dir /hdd/work/ENR/data/layer_emb \
#     --layer_num 12 \
#     --dry_run True


### Layer search
# for layer_num in {0..12}
# do
#     CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python src/inference/layer_search.py \
#         --base_model BAAI/bge-base-en-v1.5 \
#         --do_test_only True \
#         --data_dir data/msmarco \
#         --search_out_dir result/search \
#         --validation_file dev.jsonl \
#         --use_faiss True \
#         --search_split dev \
#         --q_max_len 32 \
#         --search_topk 100 \
#         --batch_size 64 \
#         --encode_save_dir /hdd/work/ENR/data/layer_emb \
#         --layer_num $layer_num
# done


### Write faiss index
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python src/inference/layer_search.py \
    --base_model BAAI/bge-base-en-v1.5 \
    --do_test_only True \
    --data_dir data/msmarco \
    --search_out_dir result/search \
    --validation_file dev.jsonl \
    --use_faiss True \
    --search_split dev \
    --q_max_len 32 \
    --search_topk 100 \
    --batch_size 64 \
    --encode_save_dir /hdd/work/ENR/data/final_emb/doc2docid