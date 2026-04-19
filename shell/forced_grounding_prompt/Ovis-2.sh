#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "Ovis-2" \
    --config configs/models.yaml \
    --batch_size 4 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
