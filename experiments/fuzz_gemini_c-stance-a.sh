#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    c-stance-a \
    zeroshot \
    results/zeroshot+gemini-2.5-flash-lite+c-stance-a.jsonl \
    --cls-model gemini-2.5-flash-lite \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300 \
    --sample-n 1000
