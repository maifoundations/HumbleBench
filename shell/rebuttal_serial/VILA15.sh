#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=2

/bk/HumbleBench/envs/env_5/bin/python \
    main.py \
    --model "VILA1.5" \
    --config configs/models.yaml \
    --batch_size 1 \
    --shuffle_nota_position \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_5/bin/python \
    main.py \
    --model "VILA1.5" \
    --config configs/models.yaml \
    --batch_size 1 \
    --use_cautious_prompt \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_5/bin/python \
    main.py \
    --model "VILA1.5" \
    --config configs/models.yaml \
    --batch_size 1 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
