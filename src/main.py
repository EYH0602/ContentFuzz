import logging
from typing import Literal

import pandas as pd
import fire

from contentfuzz.cls import StanceAnalyzer, OpenAIAnalyzer
from contentfuzz.stance_dataset import StanceDataset, load_c_stance, DATASET
from contentfuzz.evaluate import (
    EvalMetrics,
    compute_metrics,
    print_eval_metrics,
)

from contentfuzz.run import run_generation


def main(dataset_name: DATASET, output_result_path: str = "out.jsonl") -> None:
    """Main entry point to run ContentFuzz.

    Args:
        dataset: Which dataset to use ("c-stance-a" or "c-stance-b").
        split: HF split to use (default: "train").
        output_result_path: Path to save results JSONL.
    """

    # dataset only have C-STANCE for now
    dataset: StanceDataset = load_c_stance(dataset_name, "test")

    analyzer: StanceAnalyzer = OpenAIAnalyzer()
    logging.info(f"Running analysis with {analyzer.__class__.__name__}.")
    results = run_generation(
        dataset,
        analyzer,
        output_result_path=output_result_path,
    )

    eval_results: EvalMetrics = compute_metrics(pd.DataFrame(results))
    print_eval_metrics(eval_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
