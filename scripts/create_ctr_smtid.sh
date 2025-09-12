#!/bin/bash

### Sample
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src/ python src/inference/create_ctr_smtid.py \
#     --base_model BAAI/bge-base-en-v1.5 \
#     --data_dir data/msmarco \
#     --search_out_dir data/msmarco/ctr_smtid \
#     --validation_file dev.jsonl \
#     --smtid_file data/bge_base_en_v1.5/smtid_ms_full.npy \
#     --search_split dev \
#     --q_max_len 32 \
#     --search_topk 100 \
#     --batch_size 64 \
#     --encode_save_dir /home/junho/backup/junho/enr/data/bge-base/faiss_index \
#     --dry_run True


### BGE-base model
# train data
# Due to the size of train data, the file is splitted into 10 chunks
# mkdir -p data/msmarco/ctr_smtid/splitted
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src_jh/ python src_jh/inference/create_ctr_smtid.py \
    --base_model BAAI/bge-base-en-v1.5 \
    --data_dir data/msmarco \
    --search_out_dir data/msmarco/ctr_smtid \
    --validation_file train.jsonl \
    --smtid_file data/bge_base_en_v1.5/smtid_ms_full.npy \
    --search_split train \
    --q_max_len 32 \
    --search_topk 100 \
    --batch_size 64 \
    --encode_save_dir /hdd/work/ENR/data/final_emb/bge-base/faiss_index
echo Done create contrastive smtid dataset for train


# dev data
# CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src_jh/ python src_jh/inference/create_ctr_smtid.py \
#     --base_model BAAI/bge-base-en-v1.5 \
#     --data_dir data/msmarco \
#     --search_out_dir data/msmarco/ctr_smtid \
#     --validation_file dev.jsonl \
#     --smtid_file data/bge_base_en_v1.5/smtid_ms_full.npy \
#     --search_split dev \
#     --q_max_len 32 \
#     --search_topk 100 \
#     --batch_size 64 \
#     --encode_save_dir /hdd/work/ENR/data/final_emb/bge-base/faiss_index
# echo Done create contrastive smtid dataset for dev