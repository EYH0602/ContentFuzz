#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    vast \
    zeroshot \
    results/zeroshot+gemini-2.5-flash-lite+vast.jsonl \
    --cls-model gemini-2.5-flash-lite \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300