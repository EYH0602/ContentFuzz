#!/bin/bash

set -euo pipefail

schedules=("fifo" "priority" "random" "priority_random")

for schedule in "${schedules[@]}"; do
    echo "Running schedule: ${schedule}"
    uv run src/run_fuzz.py \
        sem16 \
        zeroshot \
        results/zeroshot+gemini-2.5-flash-lite+sem16.jsonl \
        --cls-model gemini-2.5-flash-lite \
        --fuzzer-model gemini-2.5-flash-lite \
        --sample-n 100 \
        --schedule "${schedule}"
done
