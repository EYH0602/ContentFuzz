import argparse
import logging
import os
import random
from typing import get_args

import pandas as pd

from contentfuzz.cls import (
    StanceAnalyzer,
    ZeroshotAnalyzer,
    COLA,
    Encoder,
    Analyzer,
)
from contentfuzz.stance_dataset import StanceDataset, load_c_stance, Dataset
from contentfuzz.evaluate import (
    EvalMetrics,
    compute_metrics,
    print_eval_metrics,
)
from contentfuzz.run import run_generation
from contentfuzz.utils import SEED, get_default_cls_output_path


def main(
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    model: str = "gemini-2.5-flash-lite",
    sample_n: int | None = None,
    output_result_path: str | None = None,
) -> None:
    """Main entry point to run Stance Analysis in ContentFuzz.

    Usage:
    python src/run_cls.py -h
    """

    if output_result_path is None:
        output_dir = os.path.abspath("results")
        os.makedirs(output_dir, exist_ok=True)
        output_result_path = get_default_cls_output_path(
            output_dir, dataset_name, analyzer_name, model
        )

    random.seed(SEED)
    # dataset only have C-STANCE for now
    dataset: StanceDataset
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            dataset = load_c_stance(dataset_name, "test")

    if sample_n is not None:
        dataset = random.sample(dataset, sample_n)

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=model)
        case "cola":
            analyzer = COLA(model=model)
        case "encoder":
            analyzer = Encoder(model=model)

    logging.info(f"Running {analyzer.__class__.__name__} with {analyzer.model}.")
    results = run_generation(
        dataset,
        analyzer,
        output_result_path=output_result_path,
    )

    eval_results: EvalMetrics = compute_metrics(pd.DataFrame(results))
    print_eval_metrics(eval_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_choices = get_args(Dataset)
    analyzer_choices = get_args(Analyzer)
    parser = argparse.ArgumentParser(
        description="Run ContentFuzz stance classification experiment."
    )
    parser.add_argument(
        "dataset_name",
        choices=dataset_choices,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "analyzer_name",
        choices=analyzer_choices,
        help="Analyzer to run.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-2.5-flash-lite",
        help="Gemini Model to use for generation.",
    )
    parser.add_argument(
        "-n",
        "--sample-n",
        dest="sample_n",
        type=int,
        help="Optional number of dataset rows to sample before running.",
    )
    parser.add_argument(
        "-o",
        "--output-result-path",
        dest="output_result_path",
        help=(
            "Optional path to store generation results JSONL; defaults to "
            "results/{analyzer}+{model}+{dataset}.jsonl."
        ),
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        analyzer_name=args.analyzer_name,
        model=args.model,
        sample_n=args.sample_n,
        output_result_path=args.output_result_path,
    )
