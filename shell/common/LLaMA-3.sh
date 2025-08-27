#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "LLaMA-3" \
    --config configs/models.yaml \
    --batch_size 16 \
    --log_dir results/common
