#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "R1-Onevision" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
