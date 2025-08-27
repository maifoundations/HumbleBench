#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

/mnt/data2/bingkui/miniconda3/envs/mulberry/bin/python\
    main.py \
    --model "Mulberry" \
    --config configs/models.yaml \
    --batch_size 4 \
    --nota_only \
    --log_dir results/nota_only
