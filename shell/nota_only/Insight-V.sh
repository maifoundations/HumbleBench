#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python \
    main.py \
    --model "Insight-V" \
    --config configs/models.yaml \
    --batch_size 1 \
    --nota_only \
    --log_dir results/nota_only
