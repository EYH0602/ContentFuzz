import fire
from contentfuzz.evaluate import load_gen_results, compute_metrics, print_eval_metrics


def main(results_file: str):
    """run evaluation on the saved JSONL generation results file"""

    df = load_gen_results(results_file)
    metrics = compute_metrics(df)

    print_eval_metrics(metrics)


if __name__ == "__main__":
    fire.Fire(main)
