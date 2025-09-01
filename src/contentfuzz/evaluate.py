from typing import TypedDict

import pandas as pd
import orjson


class GenResult(TypedDict):
    """saved generation results"""

    truth: str
    predicted: str
    rationale: str
    confidence: float | None


class EvalMetrics(TypedDict):
    """Evaluation metrics for model performance."""

    accuracy: float
    average_correct_confidence: float
    average_incorrect_confidence: float


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
    avg_correct_confidence = df[df["truth"] == df["predicted"]]["confidence"].mean()
    avg_incorrect_confidence = df[df["truth"] != df["predicted"]]["confidence"].mean()

    return {
        "accuracy": round(accuracy, 4),
        "average_correct_confidence": round(avg_correct_confidence, 4),
        "average_incorrect_confidence": round(avg_incorrect_confidence, 4),
    }


def print_eval_metrics(metrics: EvalMetrics) -> None:
    """Print evaluation metrics."""
    print(
        orjson.dumps(
            metrics,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
        ).decode()
    )


def get_correct_tasks(
    tasks: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get tasks that were correctly predicted."""

    assert len(tasks) == len(results), "Tasks and results length mismatch"

    correct_indices = results[results["truth"] == results["predicted"]].index
    return tasks.iloc[correct_indices], results.iloc[correct_indices]
