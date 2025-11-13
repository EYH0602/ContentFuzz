#!/bin/bash

set -euo pipefail

uv run src/run_fuzz.py \
    c-stance-a \
    encoder \
    results/encoder+saved_models--hfl--chinese-bert-wwm+c-stance-a.jsonl  \
    --cls-model saved_models/hfl/chinese-bert-wwm \
    --fuzzer-model gemini-2.5-flash-lite \
    --schedule priority \
    --n-iters 300