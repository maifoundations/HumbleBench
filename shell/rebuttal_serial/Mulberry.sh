#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=3

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "Mulberry" \
    --config configs/models.yaml \
    --batch_size 8 \
    --shuffle_nota_position \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "Mulberry" \
    --config configs/models.yaml \
    --batch_size 8 \
    --use_cautious_prompt \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "Mulberry" \
    --config configs/models.yaml \
    --batch_size 8 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
