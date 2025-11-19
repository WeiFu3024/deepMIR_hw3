#!/bin/bash

model_path=$1
loss=$2

# Execute the commands within the specified environment
micromamba run -n cp_transformers bash -c "\
    cd compound-word-transformer; \
    python3 workspace/uncond/cp-linear/inference_controlled.py \
        --model_path $model_path \
        --loss $loss \
        --dict_path ../Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl \
        --output_dir ../results/cpword_${loss} \
        --num_bars 32 \
        --num_samples 20 \
        --temperature 1.2 \
        --top_p 0.9 \
        --gid 0; \
    python3 workspace/uncond/cp-linear/inference_controlled.py \
        --model_path $model_path \
        --loss $loss \
        --dict_path ../Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl \
        --output_dir ../results/cpword_${loss}/continuation \
        --prompt_midi_dir ../prompt_song \
        --num_bars 32 \
        --num_samples 20 \
        --temperature 1.2 \
        --top_p 0.9 \
        --gid 0; \
    cd ..; \
    python3 render.py --midi_dir ./results/cpword_${loss}/\
        --store_dir ./results/cpword_${loss}/; \
    python3 render.py --midi_dir ./results/cpword_${loss}/continuation \
        --store_dir ./results/cpword_${loss}/continuation; \
"