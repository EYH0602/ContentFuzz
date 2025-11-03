import random
import fire
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
    fire.Fire(main)
