#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1 python main.py --do_train_only True \
    --base_model bert-base-uncased \
    --model_pretrained bert-base-uncased \
    --encoder_model bge_base_en_v1.5 \
    --exp_name bge_base_en_v1.5 \
    --num_workers 32 \
    --batch_size 256 \
    --epoch 5 \
    --train_codebook False \
    --pretrain True