#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1 python main.py --do_train_only True \
    --base_model ../pretrained_model/bge_pretrained \
    --model_pretrained ../pretrained_model/bge_pretrained \
    --encoder_model bge_base_en_v1.5 \
    --exp_name bge_pretrined \
    --num_workers 32 \
    --batch_size 256 \
    --epoch 50 \
    --train_codebook False