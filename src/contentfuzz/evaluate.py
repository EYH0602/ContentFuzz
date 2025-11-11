from typing import TypedDict, Literal

import pandas as pd
import orjson
from sklearn.metrics import f1_score

from .stance_dataset import StanceDataset
from ._types import Stance


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


class FuzzMetrics(TypedDict):
    """Evaluation metrics for fuzzing performance"""

    attack_succ_rate: float


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


def compute_fuzz_metrics(df: pd.DataFrame) -> FuzzMetrics:
    """
    Compute fuzzing metrics where success is defined as matching stances.

    A row is considered a failure if its predicted value equals "error";
    otherwise, it is a success only when the predicted stance matches the
    reference stance.
    """

    assert {"stance", "predicted"}.issubset(df.columns)

    if df.empty:
        return {"attack_succ_rate": 0.0}

    predicted = df["predicted"].astype(str)
    stance = df["stance"].astype(str)

    success_mask = stance != predicted

    attack_succ_rate = success_mask.sum() / len(df)

    return {"attack_succ_rate": round(float(attack_succ_rate), 4)}
