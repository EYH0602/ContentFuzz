#!/bin/bash

model="gemini-2.5-flash-lite"
result_path=results/encoder+saved_models--FacebookAI--roberta-base+sem16+sem16.jsonl
model_path=saved_models/FacebookAI/roberta-base+sem16
n_samples=200

set -euo pipefail

uv run src/run_fuzz.py \
    sem16 \
    encoder \
    $result_path \
    --cls-model $model_path \
    --fuzzer-model $model \
    --temperature 1.0 \
    --sample-n $n_samples \
    -ao fuzz/sem16_temp_1_gemini-2.5-flash-lite_200.jsonl 

uv run src/run_fuzz.py \
    sem16 \
    encoder \
    $result_path \
    --cls-model $model_path \
    --fuzzer-model $model \
    --sample-n $n_samples \
    -ao fuzz/sem16_temp_sched_gemini-2.5-flash-lite_200.jsonl 