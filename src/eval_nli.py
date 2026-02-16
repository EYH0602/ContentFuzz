"""Run NLI entailment/contradiction evaluation on fuzz result files."""

import argparse
import glob
import os

import orjson
import pandas as pd

from contentfuzz.evaluate import compute_nli, print_eval_metrics
from contentfuzz.stance_dataset import DatasetLangMap
from eval_fuzz import parse_dataset_from_filename


def run_nli_on_file(results_file: str) -> dict | None:
    """Run NLI evaluation on a single JSONL results file."""
    df = pd.read_json(results_file, lines=True)
    assert {"stance", "predicted", "text", "new_text"}.issubset(df.columns)

    dataset = parse_dataset_from_filename(results_file)
    lang = DatasetLangMap[dataset]

    error_series = (
        df["error"] if "error" in df.columns else pd.Series(pd.NA, index=df.index)
    )
    success_mask = df["predicted"].notna() & error_series.isna()
    n_success = int(success_mask.sum())

    if n_success == 0:
        return None

    orig_posts = df.loc[success_mask, "text"].astype(str).tolist()
    fuzzed_posts = df.loc[success_mask, "new_text"].astype(str).tolist()

    nli_result = compute_nli(orig_posts, fuzzed_posts, lang=lang)
    if nli_result is None:
        return None

    return {
        "file": os.path.basename(results_file),
        "dataset": dataset,
        "lang": lang,
        "n_success": n_success,
        "nli": nli_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NLI evaluation on ContentFuzz results."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="JSONL result files to evaluate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSONL files under fuzz/.",
    )
    args = parser.parse_args()

    files: list[str] = args.files
    if args.all:
        files = sorted(glob.glob("fuzz/**/*.jsonl", recursive=True))

    if not files:
        parser.error("No files specified. Provide file paths or use --all.")

    results = []
    for f in files:
        print(f"Processing: {f}")
        result = run_nli_on_file(f)
        if result is not None:
            results.append(result)
            print_eval_metrics(result["nli"])
        else:
            print("  (no successful rows, skipping)")
        print()

    # Print summary table
    if results:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(
            orjson.dumps(
                results,
                option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
            ).decode()
        )


if __name__ == "__main__":
    main()
