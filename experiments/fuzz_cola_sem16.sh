#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    sem16 \
    cola \
    results/cola+gemini-2.5-flash-lite+sem16.jsonl \
    --cls-model gemini-2.5-flash-lite \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300 \
    --sample-n 100
