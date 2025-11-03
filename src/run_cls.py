import logging
import os
import random

import pandas as pd
import fire

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
from contentfuzz.utils import SEED


def main(
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    model: str = "gemini-2.5-flash-lite",
    sample_n: int | None = None,
    output_result_path: str | None = None,
) -> None:
    """Main entry point to run Stance Analysis in ContentFuzz.

    Args:
        dataset_name: Which dataset to use ("c-stance-a" or "c-stance-b").
        analyzer_name: Which analyzer to use ("zero-shot", "cola").
        model: which OpenAI model to use
        output_result_path: Path to save results JSONL.
    """

    if output_result_path is None:
        os.makedirs("results", exist_ok=True)
        output_result_path = (
            f"results/{analyzer_name}+{model.replace("/", "--")}+{dataset_name}.jsonl"
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
        case "zero-shot":
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
    fire.Fire(main)
