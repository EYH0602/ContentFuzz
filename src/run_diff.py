import logging

import pandas as pd
import fire
from contentfuzz.evaluate import (
    EvalMetrics,
    compute_metrics,
    print_eval_metrics,
    load_gen_results,
    get_correct_tasks,
)


def main(input_file_path: str, output_result_path: str = "out.jsonl") -> None:
    """Main entry point to run ContentFuzz"""

    dataset = pd.read_csv(input_file_path)
    gen_results = load_gen_results(output_result_path)

    assert len(dataset) == len(gen_results), "Dataset and results length mismatch"

    eval_results: EvalMetrics = compute_metrics(gen_results)
    print_eval_metrics(eval_results)

    correct_tasks = get_correct_tasks(dataset, gen_results)
    print(len(correct_tasks))
    print(len(correct_tasks) / len(dataset))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
