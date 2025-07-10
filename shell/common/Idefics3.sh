#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

/home/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "Idefics3" \
    --config configs/models.yaml \
    --batch_size 16 \
    --log_dir results/common
