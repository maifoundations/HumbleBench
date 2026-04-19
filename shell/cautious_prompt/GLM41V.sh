#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "GLM-4.1V" \
    --config configs/models.yaml \
    --batch_size 16 \
    --use_cautious_prompt \
    --log_dir rebuttal_results
