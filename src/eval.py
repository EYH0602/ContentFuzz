import fire
from contentfuzz.evaluate import load_to_df, compute_metrics, print_eval_metrics


def main(input_results_file: str):
    """run evaluation on the saved JSONL generation results file"""

    df = load_to_df(input_results_file)
    metrics = compute_metrics(df)

    print_eval_metrics(metrics)


if __name__ == "__main__":
    fire.Fire(main)
