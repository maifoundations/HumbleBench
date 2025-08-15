#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

/mnt/data2/bingkui/miniconda3/envs/glmv/bin/python\
    main.py \
    --model "GLM-4.1V" \
    --config configs/models.yaml \
    --batch_size 16 \
    --log_dir results/noise_image \
    --use_noise_image
