from typing import TypedDict, Literal
import logging

import pandas as pd
import orjson
from sklearn.metrics import f1_score
import evaluate
import numpy as np
import matplotlib.pyplot as plt

from .stance_dataset import StanceDataset
from ._types import Stance
from .utils import Language


class GenResult(TypedDict):
    """saved generation results"""

    truth: Stance
    predicted: Stance | Literal["error"]
    confidence: float | None


class EvalMetrics(TypedDict):
    """Evaluation metrics for model performance"""

    accuracy: float
    f1: float
    average_correct_confidence: float
    average_incorrect_confidence: float


class IterationStats(TypedDict):
    """Statistics about iterations to success"""

    mean: float
    median: float
    std: float
    minimum: int
    maximum: int


class BERTScore(TypedDict):
    """BERTScore metrics"""

    precision: float
    recall: float
    f1: float


class Perplexity(TypedDict):
    """Perplexity metrics with distribution stats"""

    mean: float
    std: float
    median: float
    minimum: float
    maximum: float
    majority_mean: float | None
    majority_range: tuple[float, float]


class FuzzMetrics(TypedDict):
    """Evaluation metrics for fuzzing performance"""

    attack_succ_rate: float | None
    iters: IterationStats | None
    bertscore: BERTScore | None
    perplexity: Perplexity | None


def load_gen_results(file_path: str) -> pd.DataFrame:
    """
    Load results from a JSONL file into a pandas DataFrame.
    """
    assert file_path.endswith(".jsonl"), "Input file must be a JSONL file"
    return pd.read_json(file_path, lines=True)


def compute_metrics(df: pd.DataFrame) -> EvalMetrics:
    """
    Compute evaluation metrics from the results DataFrame.
    """

    # check if DataFrame has required columns
    assert {"truth", "predicted", "confidence"}.issubset(df.columns)

    accuracy = (df["truth"] == df["predicted"]).mean()
    # Use sklearn for macro F1; cast to string to avoid label type issues
    f1 = f1_score(
        df["truth"].astype(str),
        df["predicted"].astype(str),
        average="macro",
        zero_division=0,
    )
    avg_correct_confidence = df[df["truth"] == df["predicted"]]["confidence"].mean()
    avg_incorrect_confidence = df[df["truth"] != df["predicted"]]["confidence"].mean()

    return {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4),
        "average_correct_confidence": round(avg_correct_confidence, 4),
        "average_incorrect_confidence": round(avg_incorrect_confidence, 4),
    }


def print_eval_metrics(metrics: EvalMetrics | FuzzMetrics) -> None:
    """Print evaluation metrics."""
    print(
        orjson.dumps(
            metrics,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
        ).decode()
    )


def get_correct_tasks(
    tasks: StanceDataset, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get tasks that were correctly predicted."""

    assert len(tasks) == len(results), "Tasks and results length mismatch"

    correct_indices = results[results["truth"] == results["predicted"]].index
    tasks_df = pd.DataFrame(tasks)
    return tasks_df.iloc[correct_indices], results.iloc[correct_indices]


def compute_perplexity(
    posts: list[str], alpha: float = 0.05, fast: bool = False
) -> Perplexity | None:
    """
    Compute the perplexity between original and fuzzed posts.
    """

    perplexity_metric = evaluate.load("perplexity", module_type="measurement")
    model_id = "gpt2" if fast else "google/gemma-3-1b-pt"
    batch_size = 32 if fast else 8
    logging.info(f"Computing perplexity using {model_id} with batch size {batch_size}")
    fuzzed_results = perplexity_metric.compute(
        data=posts,
        model_id=model_id,
        batch_size=batch_size,
    )

    if not fuzzed_results or "perplexities" not in fuzzed_results:
        return None
    # Cast to numpy array so we can use vectorized percentile/masking ops safely
    perplexities = np.asarray(fuzzed_results["perplexities"], dtype=float)

    # majority mean, (1 - alpha)% truncated
    lower = np.percentile(perplexities, alpha / 2 * 100)
    upper = np.percentile(perplexities, (1 - alpha / 2) * 100)
    mask = (perplexities >= lower) & (perplexities <= upper)
    majority_mean = (
        round(float(np.mean(perplexities[mask])), 4) if np.any(mask) else None
    )
    majority_range = (round(float(lower), 4), round(float(upper), 4))
    return {
        "mean": round(float(np.mean(perplexities)), 4),
        "std": round(float(np.std(perplexities)), 4),
        "median": round(float(np.median(perplexities)), 4),
        "minimum": round(float(np.min(perplexities)), 4),
        "maximum": round(float(np.max(perplexities)), 4),
        "majority_mean": majority_mean,
        "majority_range": majority_range,
    }


def compute_fuzz_metrics(
    df: pd.DataFrame, lang: Language = "en", fast: bool = False
) -> FuzzMetrics:
    """
    Compute fuzzing metrics where success is defined as matching stances.

    A row is considered a failure if its predicted value equals "error";
    otherwise, it is a success only when the predicted stance matches the
    reference stance.
    """

    assert {"stance", "predicted", "iteration"}.issubset(df.columns)

    if df.empty:
        return {
            "attack_succ_rate": None,
            "iters": None,
            "bertscore": None,
            "perplexity": None,
        }

    # success rate
    predicted = df["predicted"].astype(str)
    stance = df["stance"].astype(str)
    success_mask = stance != predicted
    attack_succ_rate = success_mask.sum() / len(df)

    # iterations to success
    # iteration counts are zero-indexed; add 1 before computing stats
    success_iterations = (
        pd.to_numeric(df.loc[success_mask, "iteration"], errors="coerce") + 1
    ).dropna()

    iter_stats: IterationStats | None = None
    if not success_iterations.empty:
        iterations_int = success_iterations.astype(int)
        iter_stats = {
            "mean": round(float(iterations_int.mean()), 4),
            "median": round(float(iterations_int.median()), 4),
            "std": round(float(iterations_int.std(ddof=0)), 4),
            "minimum": int(iterations_int.min()),
            "maximum": int(iterations_int.max()),
        }

    # BERTScore
    orig_correct_posts = df.loc[success_mask, "text"].astype(str).tolist()
    fuzzed_correct_posts = df.loc[success_mask, "new_text"].astype(str).tolist()
    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=fuzzed_correct_posts,
        references=orig_correct_posts,
        lang=lang,
    )

    # Perplexity Ratio
    ppl = compute_perplexity(fuzzed_correct_posts, fast=fast)

    return {
        "attack_succ_rate": round(float(attack_succ_rate), 4),
        "iters": iter_stats,
        "bertscore": (
            {
                "precision": round(np.mean(bertscore_results["precision"]), 4),
                "recall": round(np.mean(bertscore_results["recall"]), 4),
                "f1": round(np.mean(bertscore_results["f1"]), 4),
            }
            if bertscore_results
            else None
        ),
        "perplexity": ppl,
    }
