#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

/home/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "Ovis-2" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/noise_image \
    --use_noise_image
