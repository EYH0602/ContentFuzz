import argparse
import random
import os
from contentfuzz.evaluate import (
    load_gen_results,
    print_eval_metrics,
    compute_fuzz_metrics,
)
from contentfuzz.utils import SEED
from contentfuzz.stance_dataset import DatasetLangMap, Dataset, is_dataset


def parse_dataset_from_filename(filename: str) -> Dataset:
    """extract dataset name from results filename"""
    base_name = os.path.basename(filename)
    file_name = os.path.splitext(base_name)[0]
    cls_id = file_name.split("=")[0]
    dataset = cls_id.split("+")[-1]
    assert is_dataset(
        dataset
    ), f"Could not parse valid dataset from filename: {filename}"
    return dataset


def main(results_file: str, sample_n: int | None = None, fast: bool = False) -> None:
    """run evaluation on the saved JSONL generation results file"""

    df = load_gen_results(results_file)

    if sample_n is not None:
        random.seed(SEED)
        df = df.sample(sample_n, random_state=SEED)

    dataset_name = parse_dataset_from_filename(results_file)
    metrics = compute_fuzz_metrics(
        df,
        lang=DatasetLangMap[dataset_name],
        fast=fast,
        include_bertscore=True,
        include_perplexity=True,
        include_mauve=True,
    )
    print_eval_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation on saved ContentFuzz generation results."
    )
    parser.add_argument(
        "results_file", help="Path to JSONL results produced by run_cls."
    )
    parser.add_argument(
        "-n",
        "--sample-n",
        dest="sample_n",
        type=int,
        default=None,
        help="Optional number of dataset rows to sample before running.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode for evaluation.",
    )
    args = parser.parse_args()
    main(results_file=args.results_file, sample_n=args.sample_n, fast=args.fast)
