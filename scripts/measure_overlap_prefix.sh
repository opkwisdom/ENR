#!/bin/bash

### Sample
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src_jh/ python src_jh/inference/measure_overlap_prefix.py \
    --search_out_dir result/hard_negs