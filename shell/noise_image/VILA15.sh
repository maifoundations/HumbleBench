#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

/home/bingkui/miniconda3/envs/vila/bin/python\
    main.py \
    --model "VILA1.5" \
    --config configs/models.yaml \
    --batch_size 1 \
    --log_dir results/noise_image \
    --use_noise_image
