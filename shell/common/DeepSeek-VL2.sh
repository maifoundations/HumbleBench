#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

/home/bingkui/miniconda3/envs/deepseekvl/bin/python\
    main.py \
    --model "DeepSeek-VL2" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
