import logging

import pandas as pd
import fire

from contentfuzz.cls import StanceAnalyzer, OpenAIAnalyzer, COLA
from contentfuzz.datasets import load_c_stance
from contentfuzz.evaluate import (
    EvalMetrics,
    compute_metrics,
    print_eval_metrics,
)

from contentfuzz.run import run_generation


def main(input_file_path: str, output_result_path: str = "out.jsonl") -> None:
    """Main entry point to run ContentFuzz"""

    dataset = load_c_stance(input_file_path)
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
