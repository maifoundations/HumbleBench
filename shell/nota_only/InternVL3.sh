#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "InternVL3" \
    --config configs/models.yaml \
    --batch_size 4 \
    --nota_only \
    --log_dir results/nota_only
