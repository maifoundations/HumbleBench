#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

/mnt/data2/bingkui/miniconda3/envs/vila/bin/python\
    main.py \
    --model "VILA1.5" \
    --config configs/models.yaml \
    --batch_size 1 \
    --nota_only \
    --log_dir results/nota_only
