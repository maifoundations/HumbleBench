#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/home/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "Qwen2.5-VL" \
    --config configs/models.yaml \
    --batch_size 16 \
    --log_dir results/common
