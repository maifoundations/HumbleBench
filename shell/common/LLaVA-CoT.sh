#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

/mnt/data2/bingkui/miniconda3/envs/qwenvl25/bin/python\
    main.py \
    --model "LLaVA-CoT" \
    --config configs/models.yaml \
    --batch_size 4 \
    --log_dir results/common
