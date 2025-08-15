#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

/mnt/data2/bingkui/miniconda3/envs/XCD/bin/python\
    main.py \
    --model "Cambrian" \
    --config configs/models.yaml \
    --batch_size 4 \
    --nota_only \
    --log_dir results/nota_only
