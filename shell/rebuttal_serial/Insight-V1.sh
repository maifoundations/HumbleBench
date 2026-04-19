#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1

# /bk/HumbleBench/envs/env_2/bin/python \
#     main.py \
#     --model "Insight-V" \
#     --config configs/models.yaml \
#     --batch_size 2 \
#     --shuffle_nota_position \
#     --log_dir rebuttal_results

/bk/HumbleBench/envs/env_2/bin/python \
    main.py \
    --model "Insight-V" \
    --config configs/models.yaml \
    --batch_size 2 \
    --use_cautious_prompt \
    --log_dir rebuttal_results

# /bk/HumbleBench/envs/env_2/bin/python \
#     main.py \
#     --model "Insight-V" \
#     --config configs/models.yaml \
#     --batch_size 2 \
#     --use_noise_image \
#     --use_forced_grounding_prompt \
#     --log_dir rebuttal_results
