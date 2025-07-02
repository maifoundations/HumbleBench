#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

/home/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "R1-VL" \
    --config configs/models.yaml \
    --batch_size 8 \
    --log_dir results/noise_image \
    --use_noise_image 
