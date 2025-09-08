#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1 python main.py --do_train_only True \
    --base_model bert-base-uncased \
    --model_pretrained ../pretrained_model/bge_pretrained  \
    --encoder_model bge_base_en_v1.5 \
    --exp_name bge_base_en_v1.5 \
    --num_workers 32 \
    --batch_size 256 \
    --val_check_interval 0.25 \
    --epoch 3 \
    --train_codebook True \
    --train_encoder False \
    --pretrain True \
    --pretrain_path pretrain_with_train.pickle