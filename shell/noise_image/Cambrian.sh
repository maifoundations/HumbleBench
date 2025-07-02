#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

/home/bingkui/miniconda3/envs/XCD/bin/python\
    main.py \
    --model "Cambrian" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/noise_image \
    --use_noise_image
