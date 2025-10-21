#!/usr/bin/env bash
set -euo pipefail

# Run ContentFuzz for both C-STANCE datasets using the new main.py interface.

MODEL_SUFFIX="${MODEL_SUFFIX:-gpt-4.1-nano}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

datasets=(
  "c-stance-a"
  "c-stance-b"
)

echo "Running ContentFuzz over C-STANCE datasets..."

for ds in "${datasets[@]}"; do
  case "$ds" in
    c-stance-a) tag="A" ;;
    c-stance-b) tag="B" ;;
    *) tag="$(echo "$ds" | tr '[:lower:]' '[:upper:]')" ;;
  esac

  out="${RESULTS_DIR}/c_stance_${tag}_${MODEL_SUFFIX}.jsonl"

  echo "\n==> Dataset: $ds"
  echo "Output:  $out"
  uv run src/main.py "$ds" --output_result_path "$out"
done

echo "\nAll datasets completed."
