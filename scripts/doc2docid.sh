#!/bin/bash

### Sample
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src_jh/ python src_jh/doc2docid_main.py \
    --base_model bert-base-uncased \
    --model_pretrained bert-base-uncased \
    --exp_name enr \
    --data_dir data/msmarco/pretrain \
    --codebook_dir data/bge_base_en_v1.5 \
    --num_workers 32 \
    --batch_size 128 \
    --epoch 10 \
    --train_encoder True \
    --smtid_layer "[2, 4, 6, 8, 9, 10, 11, 12]" \
    --prefix_accuracy "[1, 5, 10, 20]" \
    --checkpoint_monitor valid/loss


# CUDA_VISIBLE_DEVICES=0,1 python main.py --do_train_only True \
#     --base_model bert-base-uncased \
#     --model_pretrained bert-base-uncased \
#     --encoder_model bge_base_en_v1.5 \
#     --exp_name bge_base_en_v1.5 \
#     --num_workers 32 \
#     --batch_size 256 \
#     --epoch 5 \
#     --train_codebook False \
#     --pretrain True