#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    sem16 \
    encoder \
    results/encoder+saved_models--google-bert--bert-base-cased--sem16+sem16.jsonl \
    --cls-model saved_models/google-bert/bert-base-cased/sem16 \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300