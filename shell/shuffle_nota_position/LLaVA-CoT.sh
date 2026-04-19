#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "LLaVA-CoT" \
    --config configs/models.yaml \
    --batch_size 4 \
    --shuffle_nota_position \
    --log_dir rebuttal_results
