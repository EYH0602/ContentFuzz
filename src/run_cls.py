import argparse
import logging
import os
import random
from typing import get_args

import pandas as pd

from contentfuzz.cls import (
    COLA,
    Analyzer,
    Encoder,
    StanceAnalyzer,
    ZeroshotAnalyzer,
)
from contentfuzz.evaluate import (
    EvalMetrics,
    compute_metrics,
    print_eval_metrics,
)
from contentfuzz.run import run_batch_generation, run_generation
from contentfuzz.stance_dataset import (
    Dataset,
    DatasetLangMap,
    StanceDataset,
    load_c_stance,
    load_sem16,
    load_vast,
)
from contentfuzz.utils import SEED, get_default_cls_output_path, get_skip_cnt


def main(
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    model: str = "gemini-2.5-flash-lite",
    sample_n: int | None = None,
    output_result_path: str | None = None,
    batch: int = 1,
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
        case "sem16":
            dataset = load_sem16("test")
        case "vast":
            dataset = load_vast("test")
    lang = DatasetLangMap[dataset_name]

    skip_count = get_skip_cnt(output_result_path)
    dataset = dataset[skip_count:]
    if skip_count > 0:
        logging.info(
            f"Found {skip_count} results in {output_result_path}, skipping them..."
        )
    if sample_n is not None:
        dataset = random.sample(dataset, sample_n)

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=model)
        case "cola":
            analyzer = COLA(model=model, language=lang)
        case "encoder":
            analyzer = Encoder(model=model)

    logging.info(f"Running {analyzer.__class__.__name__} with {analyzer.model}.")
    results = run_batch_generation(
        dataset,
        analyzer,
        batch_size=batch,
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
    parser.add_argument(
        "-b",
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="If set, run generation in batches of the given size.",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        analyzer_name=args.analyzer_name,
        model=args.model,
        sample_n=args.sample_n,
        output_result_path=args.output_result_path,
        batch=args.batch,
    )
