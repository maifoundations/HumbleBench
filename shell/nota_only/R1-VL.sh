#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "R1-VL" \
    --config configs/models.yaml \
    --batch_size 8 \
    --nota_only \
    --log_dir results/nota_only
