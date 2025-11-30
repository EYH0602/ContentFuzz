#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    vast \
    encoder \
    results/encoder+saved_models--FacebookAI--roberta-base+vast+vast.jsonl \
    --cls-model saved_models/FacebookAI/roberta-base+vast \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300