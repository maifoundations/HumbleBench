#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

/home/bingkui/miniconda3/envs/mulberry/bin/python\
    main.py \
    --model "Mulberry" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
