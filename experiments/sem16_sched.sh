#!/bin/bash

set -euo pipefail

schedules=("fifo" "random" "priority_random")

for schedule in "${schedules[@]}"; do
    echo "Running schedule: ${schedule}"

    uv run src/run_fuzz.py \
        sem16 \
        encoder \
        results/encoder+saved_models--FacebookAI--roberta-base--sem16+sem16.jsonl \
        --cls-model saved_models/FacebookAI/roberta-base/sem16 \
        --fuzzer-model gemini-2.5-flash-lite \
        --sample-n 200 \
        --n-iters 300 \
        --schedule "${schedule}"
done
