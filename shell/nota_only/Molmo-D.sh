#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

/mnt/data2/bingkui/miniconda3/envs/molmo/bin/python \
    main.py \
    --model "Molmo-D" \
    --config configs/models.yaml \
    --batch_size 2 \
    --nota_only \
    --log_dir results/nota_only
