#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_6/bin/python \
    main.py \
    --model "R1-Onevision" \
    --config configs/models.yaml \
    --batch_size 4 \
    --use_cautious_prompt \
    --log_dir rebuttal_results
