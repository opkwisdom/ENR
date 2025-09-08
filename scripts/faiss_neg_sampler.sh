#!/bin/bash

### Sample
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src/ python src/inference/faiss_neg_sampler.py \
    --base_model BAAI/bge-base-en-v1.5 \
    --do_test_only True \
    --data_dir data/msmarco \
    --search_out_dir result/hard_negs \
    --validation_file dev.jsonl \
    --smtid_file data/bge_base_en_v1.5/smtid_ms_full.npy \
    --use_faiss True \
    --search_split dev \
    --q_max_len 32 \
    --search_topk 100 \
    --batch_size 64 \
    --encode_save_dir /hdd/work/ENR/data/layer_emb/12/faiss_index \
    --layer_num 12 \
    --dry_run True