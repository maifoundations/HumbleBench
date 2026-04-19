#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_5/bin/python \
    main.py \
    --model "VILA1.5" \
    --config configs/models.yaml \
    --batch_size 1 \
    --shuffle_nota_position \
    --log_dir rebuttal_results
