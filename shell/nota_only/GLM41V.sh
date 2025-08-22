#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

/mnt/data2/bingkui/miniconda3/envs/glmv/bin/python \
    main.py \
    --model "GLM-4.1V" \
    --config configs/models.yaml \
    --batch_size 16 \
    --nota_only \
    --log_dir results/nota_only
