#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="C-STANCE/c_stance_dataset/subtaskB"
MODEL_SUFFIX="gpt-4.1-nano"
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "Running c_stance_dataset/subtaskB subtasks one by one..."

for dir in "$BASE_DIR"/*; do
  [[ -d "$dir" ]] || continue

  input_csv="$dir/raw_test_all_onecol.csv"
  if [[ ! -f "$input_csv" ]]; then
    echo "Skipping $(basename "$dir"): input CSV not found"
    continue
  fi

  name="$(basename "$dir")"
  output_file="${RESULTS_DIR}/c_stance_B_${name}_${MODEL_SUFFIX}.jsonl"

  echo "\n==> Processing: $name"
  echo "Input:  $input_csv"
  echo "Output: $output_file"
  uv run src/main.py -i "$input_csv" -o "$output_file"
done

echo "\nAll subtasks completed."
