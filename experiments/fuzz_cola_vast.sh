#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    vast \
    cola \
    results/cola+gemini-2.5-flash-lite+vast.jsonl \
    --cls-model gemini-2.5-flash-lite \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300 \
    --sample-n 100
