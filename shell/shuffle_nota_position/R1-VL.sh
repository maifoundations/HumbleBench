#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "R1-VL" \
    --config configs/models.yaml \
    --batch_size 8 \
    --shuffle_nota_position \
    --log_dir rebuttal_results
