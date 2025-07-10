#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

/home/bingkui/miniconda3/envs/glmv/bin/python\
    main.py \
    --model "GLM-4.1V" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
