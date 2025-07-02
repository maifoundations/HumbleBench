#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

/home/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "Gemma-3" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
