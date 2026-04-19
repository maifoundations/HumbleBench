#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "LLaVA-Next" \
    --config configs/models.yaml \
    --batch_size 16 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
