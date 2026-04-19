#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "R1-VL" \
    --config configs/models.yaml \
    --batch_size 16 \
    --shuffle_nota_position \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "R1-VL" \
    --config configs/models.yaml \
    --batch_size 16 \
    --use_cautious_prompt \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "R1-VL" \
    --config configs/models.yaml \
    --batch_size 16 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
