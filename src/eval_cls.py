import argparse
import random
from contentfuzz.evaluate import load_gen_results, compute_metrics, print_eval_metrics
from contentfuzz.utils import SEED


def main(results_file: str, sample_n: int | None = None):
    """run evaluation on the saved JSONL generation results file"""

    df = load_gen_results(results_file)

    if sample_n is not None:
        random.seed(SEED)
        df = df.sample(sample_n, random_state=SEED)

    metrics = compute_metrics(df)

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
        help="Optional number of dataset rows to sample before running.",
    )
    args = parser.parse_args()
    main(results_file=args.results_file, sample_n=args.sample_n)
