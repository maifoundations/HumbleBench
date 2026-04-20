#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "Insight-V" \
    --config configs/models.yaml \
    --batch_size 2 \
    --log_dir rebuttal_results
