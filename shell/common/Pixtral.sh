#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

/home/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "Pixtral" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
