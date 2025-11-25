from typing import TypedDict, Literal

import pandas as pd
import orjson
from sklearn.metrics import f1_score
import evaluate
import numpy as np

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


class PerplexityRatio(TypedDict):
    """Perplexity ratio metrics"""

    original: float
    fuzzed: float
    ratio: float


class FuzzMetrics(TypedDict):
    """Evaluation metrics for fuzzing performance"""

    attack_succ_rate: float | None
    iters: IterationStats | None
    bertscore: BERTScore | None
    perplexity: PerplexityRatio | None


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


def compute_perplexity_ratio(
    orig_posts: list[str], fuzzed_posts: list[str]
) -> PerplexityRatio | None:
    """
    Compute the perplexity ratio between original and fuzzed posts.
    """

    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    model_id = "gpt2"
    batch_size = 16
    orig_results = perplexity_metric.compute(
        predictions=orig_posts,
        model_id=model_id,
        batch_size=batch_size,
    )
    fuzzed_results = perplexity_metric.compute(
        predictions=fuzzed_posts,
        model_id=model_id,
        batch_size=batch_size,
    )
    if fuzzed_results is None or orig_results is None:
        return None

    ratios = [
        fuzzed / orig
        for fuzzed, orig in zip(
            fuzzed_results["perplexities"], orig_results["perplexities"]
        )
        if orig > 0
    ]

    return {
        "original": round(float(orig_results["mean_perplexity"]), 4),
        "fuzzed": round(float(fuzzed_results["mean_perplexity"]), 4),
        "ratio": round(float(np.mean(ratios)), 4),
    }


def compute_fuzz_metrics(df: pd.DataFrame, lang: Language = "en") -> FuzzMetrics:
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
    pplr = compute_perplexity_ratio(
        orig_posts=orig_correct_posts,
        fuzzed_posts=fuzzed_correct_posts,
    )

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
        "perplexity": pplr,
    }
