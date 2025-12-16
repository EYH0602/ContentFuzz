"""Aggregate semantic metrics by mutation iteration for fuzzing outputs.

This script loads a single JSONL file produced by `run_fuzz.py` and
computes per-iteration averages (with standard deviation and standard error)
for semantic integrity metrics of successful mutations. Use the resulting CSV
to drive plots that show how iteration counts correlate with semantic drift.
"""
# type: ignore
# pylint: disable

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import evaluate
import numpy as np
import pandas as pd

from contentfuzz.evaluate import load_gen_results
from contentfuzz.utils import Language


def get_success_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that contain a successful mutation (no error and a prediction)."""
    error_series = (
        df["error"] if "error" in df.columns else pd.Series(pd.NA, index=df.index)
    )
    success_mask = df["predicted"].notna() & error_series.isna()
    success_df = df.loc[success_mask].copy()
    if "iteration" not in success_df.columns:
        msg = "Results are missing the 'iteration' column written by run_fuzz.py."
        raise KeyError(msg)
    success_df["iteration_num"] = success_df["iteration"].astype(int) + 1
    return success_df


def attach_bertscore(df: pd.DataFrame, lang: Language) -> pd.DataFrame:
    """Compute per-sample BERTScore (F1 only) and attach column."""
    predictions = df["new_text"].astype(str).tolist()
    references = df["text"].astype(str).tolist()
    metric = evaluate.load("bertscore")
    logging.info("Computing BERTScore for %d samples (lang=%s)", len(predictions), lang)
    scores = metric.compute(predictions=predictions, references=references, lang=lang)
    scored_df = df.copy()
    scored_df["bertscore_f1"] = scores["f1"]
    return scored_df


def _metric_stats(series: pd.Series) -> dict[str, float]:
    """Compute mean, std, and standard error for a metric series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "sem": np.nan}
    std = float(np.std(values, ddof=0))
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "sem": float(std / np.sqrt(values.size)),
    }


def summarize_by_iteration(
    df: pd.DataFrame, metric_columns: Iterable[str]
) -> pd.DataFrame:
    """Aggregate metrics per iteration with error bars (std and sem)."""
    rows: list[dict[str, float | int]] = []
    for iteration, group in df.groupby("iteration_num"):
        row: dict[str, float | int] = {"iteration": int(iteration), "count": len(group)}
        for col in metric_columns:
            if col not in group.columns:
                continue
            stats = _metric_stats(group[col])
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
            row[f"{col}_sem"] = stats["sem"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average semantic metrics w.r.t. mutation iteration for a fuzzing run. "
            "Outputs are ready for plotting with error bars."
        )
    )
    parser.add_argument(
        "input",
        help="Single fuzzing result JSONL file produced by run_fuzz.py.",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language code for semantic metrics (passed to BERTScore).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Where to write the aggregated metrics CSV. Defaults to "
            "`plots/<input_stem>_iteration_metrics.csv`."
        ),
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Disable BERTScore computation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    jsonl_path = Path(args.input)
    df = load_gen_results(str(jsonl_path))
    success_df = get_success_rows(df)
    if success_df.empty:
        msg = "No successful mutations found; nothing to summarize."
        raise SystemExit(msg)

    metric_columns: list[str] = []
    if args.skip_bertscore:
        raise SystemExit("BERTScore is disabled; no metrics left to compute.")

    success_df = attach_bertscore(success_df, lang=args.lang)  # type: ignore[arg-type]
    metric_columns.append("bertscore_f1")

    summary = summarize_by_iteration(success_df, metric_columns)

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        default_name = f"{Path(args.input).stem}_iteration_metrics.csv"
        output_csv = Path("plots") / default_name
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    logging.info("Wrote aggregated metrics to %s", output_csv)


if __name__ == "__main__":
    main()
