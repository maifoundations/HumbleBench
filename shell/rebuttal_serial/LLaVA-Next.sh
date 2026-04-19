#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "LLaVA-Next" \
    --config configs/models.yaml \
    --batch_size 32 \
    --shuffle_nota_position \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "LLaVA-Next" \
    --config configs/models.yaml \
    --batch_size 32 \
    --use_cautious_prompt \
    --log_dir rebuttal_results

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "LLaVA-Next" \
    --config configs/models.yaml \
    --batch_size 32 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
