#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "LLaVA-Next" \
    --config configs/models.yaml \
    --batch_size 16 \
    --log_dir results/noise_image \
    --use_noise_image
