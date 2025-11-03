#!/bin/bash

uv run src/run_fuzz.py \
    c-stance-a \
    encoder \
    results/encoder+saved_models--hfl--chinese-macbert-base+c-stance-a.jsonl \
    --cls-model saved_models/hfl/chinese-macbert-base \
    --fuzzer-model gemini-2.5-flash-lite