#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1

# /bk/HumbleBench/envs/env_1/bin/python \
#     main.py \
#     --model "GLM-4.1V" \
#     --config configs/models.yaml \
#     --batch_size 8 \
#     --shuffle_nota_position \
#     --log_dir rebuttal_results

# /bk/HumbleBench/envs/env_1/bin/python \
#     main.py \
#     --model "GLM-4.1V" \
#     --config configs/models.yaml \
#     --batch_size 8 \
#     --use_cautious_prompt \
#     --log_dir rebuttal_results

/bk/HumbleBench/envs/env_1/bin/python \
    main.py \
    --model "GLM-4.1V" \
    --config configs/models.yaml \
    --batch_size 8 \
    --use_noise_image \
    --use_forced_grounding_prompt \
    --log_dir rebuttal_results
