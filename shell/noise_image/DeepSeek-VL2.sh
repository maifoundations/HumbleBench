#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

/mnt/data2/bingkui/miniconda3/envs/deepseekvl/bin/python\
    main.py \
    --model "DeepSeek-VL2" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/noise_image \
    --use_noise_image
