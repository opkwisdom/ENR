#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python main.py \
    --base_model ../pretrained_model/bge_pretrained \
    --model_pretrained ../pretrained_model/bge_pretrained \
    --encoder_model bge_base_en_v1.5 \
    --model_load_ckpt_pth ./checkpoints/bge_base_en_v1.5-ep10-lr5e-05-bsz256/ckpt-epoch=001-val_loss=6.45284.ckpt \
    --exp_name bge_pretrined \
    --num_workers 32 \
    --batch_size 16 \
    --epoch 50 \
    --train_codebook False \
    --do_test_only True 