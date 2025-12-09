from typing import TypedDict, Literal
import logging

import pandas as pd
import orjson
from sklearn.metrics import f1_score
import evaluate
import numpy as np
import mauve
from transformers import AutoTokenizer

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


class PerplexityRatio(TypedDict):
    """Perplexity ratio metrics"""

    orig: Perplexity | None
    fuzz: Perplexity | None
    ratio_of_means: float | None
    mean_of_ratios: float | None


class FuzzMetrics(TypedDict):
    """Evaluation metrics for fuzzing performance"""

    attack_succ_rate: float | None
    iters: IterationStats | None
    bertscore: BERTScore | None
    perplexity: PerplexityRatio | None
    mauve: float | None


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


def get_majority_mean(
    data: np.ndarray, alpha: float
) -> tuple[float | None, tuple[float, float]]:
    """Compute the majority mean with (1 - alpha)% truncation.
    Returns:
    - majority_mean: mean of data within the (1 - alpha)% range
    - majority_range: (lower_bound, upper_bound) of the (1 - alpha)% range
    """
    lower = np.percentile(data, alpha / 2 * 100)
    upper = np.percentile(data, (1 - alpha / 2) * 100)
    mask = (data >= lower) & (data <= upper)
    majority_mean = round(float(np.mean(data[mask])), 4) if np.any(mask) else None
    majority_range = (round(float(lower), 4), round(float(upper), 4))
    return majority_mean, majority_range


def compute_perplexity(
    posts: list[str],
    alpha: float = 0.05,
    max_tokens: int | None = None,
) -> tuple[Perplexity, np.ndarray] | None:
    """
    Compute the perplexity between original and fuzzed posts.
    """

    perplexity_metric = evaluate.load("perplexity", module_type="measurement")
    model_id = "google/gemma-3-1b-pt"
    batch_size = 32

    logging.info(
        f"Processing posts for perplexity computation, max_tokens={max_tokens}"
    )
    processed_posts = posts
    if max_tokens is not None and max_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def _truncate(text: str) -> str:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_tokens:
                return text
            decoded = tokenizer.decode(
                token_ids[:max_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return str(decoded)

        processed_posts = [_truncate(post) for post in posts]

    logging.info(f"Computing perplexity using {model_id} with batch size {batch_size}")
    fuzzed_results = perplexity_metric.compute(
        data=processed_posts,
        model_id=model_id,
        batch_size=batch_size,
    )

    if not fuzzed_results or "perplexities" not in fuzzed_results:
        return None
    # Cast to numpy array so we can use vectorized percentile/masking ops safely
    perplexities = np.asarray(fuzzed_results["perplexities"], dtype=float)

    # majority mean, (1 - alpha)% truncated
    majority_mean, majority_range = get_majority_mean(perplexities, alpha)
    return {
        "mean": round(float(np.mean(perplexities)), 4),
        "std": round(float(np.std(perplexities)), 4),
        "median": round(float(np.median(perplexities)), 4),
        "minimum": round(float(np.min(perplexities)), 4),
        "maximum": round(float(np.max(perplexities)), 4),
        "majority_mean": majority_mean,
        "majority_range": majority_range,
    }, perplexities


def compute_perplexity_ratio(
    orig_posts: list[str],
    fuzzed_posts: list[str],
    alpha: float = 0.05,
    max_tokens: int | None = None,
) -> PerplexityRatio | None:
    """
    Compute the perplexity ratio between original and fuzzed posts.

    Args:
        orig_posts (list[str]): List of original posts.
        fuzzed_posts (list[str]): List of fuzzed posts.
        alpha (float, optional): Trimming parameter for majority mean. Defaults to 0.05.
        fast (bool, optional): If True, use fast computation. Defaults to False.
        max_tokens (int | None, optional): Maximum number of tokens to consider per post; posts will be truncated if they exceed this limit. Defaults to None.
    """

    orig_ppl, orig_perplexities = None, None
    fuzzed_ppl, fuzzed_perplexities = None, None
    ratio_of_means = None
    mean_of_ratios = None
    ratios = None
    orig_ppl_results = compute_perplexity(
        orig_posts,
        alpha=alpha,
        max_tokens=max_tokens,
    )

    if orig_ppl_results:
        orig_ppl, orig_perplexities = orig_ppl_results

    fuzzed_ppl_results = compute_perplexity(
        fuzzed_posts,
        alpha=alpha,
        max_tokens=max_tokens,
    )

    if fuzzed_ppl_results:
        fuzzed_ppl, fuzzed_perplexities = fuzzed_ppl_results

    # ratio of means
    # if there is majority mean, use that
    if orig_ppl is not None and fuzzed_ppl is not None:
        if (
            orig_ppl["majority_mean"] is not None
            and fuzzed_ppl["majority_mean"] is not None
        ):
            ratio_of_means = round(
                float(fuzzed_ppl["majority_mean"] / orig_ppl["majority_mean"]), 4
            )
        else:
            ratio_of_means = round(float(fuzzed_ppl["mean"] / orig_ppl["mean"]), 4)

    # mean of ratios
    if orig_perplexities is not None and fuzzed_perplexities is not None:
        ratios = fuzzed_perplexities / orig_perplexities
        # take majority mean of ratios with alpha trimming
        mean_of_ratios, _ = get_majority_mean(ratios, alpha)
        if mean_of_ratios is None:
            mean_of_ratios = round(float(np.mean(ratios)), 4)

    return {
        "orig": orig_ppl,
        "fuzz": fuzzed_ppl,
        "ratio_of_means": ratio_of_means,
        "mean_of_ratios": mean_of_ratios,
    }


def compute_mauve(
    orig_posts: list[str],
    fuzz_posts: list[str],
    lang: Language = "en",
) -> float | None:
    """compute mauve score"""

    model_id: str
    match lang:
        case "en":
            model_id = "google-bert/bert-base-uncased"
        case "zh":
            model_id = "hfl/chinese-bert-wwm"
    batch_size = 8
    logging.info(f"Computing mauve using {model_id} with batch size {batch_size}")
    try:
        out = mauve.compute_mauve(
            p_text=orig_posts,
            q_text=fuzz_posts,
            mauve_scaling_factor=1,
            batch_size=batch_size,
            max_text_length=512,  # bert model max seq length
            device_id=0,
            featurize_model_name=model_id,
        )
    except Exception as e:  # pylint: disable=W0718
        logging.error(f"Error computing mauve: {e}")
        return None

    return round(float(out.mauve), 4)


def compute_bertscore(
    orig_posts: list[str],
    fuzzed_posts: list[str],
    lang: Language = "en",
) -> BERTScore | None:
    """compute bertscore between original and fuzzed posts"""

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=fuzzed_posts,
        references=orig_posts,
        lang=lang,
    )
    if not bertscore_results:
        return None

    return {
        "precision": round(float(np.mean(bertscore_results["precision"])), 4),
        "recall": round(float(np.mean(bertscore_results["recall"])), 4),
        "f1": round(float(np.mean(bertscore_results["f1"])), 4),
    }


def compute_fuzz_metrics(  # pylint: disable=R0913,R0914,R0917
    df: pd.DataFrame,
    lang: Language = "en",
    include_bertscore: bool = False,
    include_perplexity: bool = False,
    include_mauve: bool = False,
) -> FuzzMetrics:
    """
    Compute fuzzing metrics.

    A row counts as a successful fuzz attempt when it produced a prediction
    (`predicted` is not NaN) and did not record an error (`error` is NaN).
    """

    assert {"stance", "predicted", "iteration"}.issubset(df.columns)

    if df.empty:
        return {
            "attack_succ_rate": None,
            "iters": None,
            "bertscore": None,
            "perplexity": None,
            "mauve": None,
        }

    error_series = (
        df["error"] if "error" in df.columns else pd.Series(pd.NA, index=df.index)
    )

    success_mask = df["predicted"].notna() & error_series.isna()

    # success rate
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

    orig_success_posts = df.loc[success_mask, "text"].astype(str).tolist()
    fuzzed_success_posts = df.loc[success_mask, "new_text"].astype(str).tolist()

    # BERTScore
    bertscore: BERTScore | None = None
    if include_bertscore:
        bertscore = compute_bertscore(
            orig_success_posts,
            fuzzed_success_posts,
            lang=lang,
        )
    # Perplexity Ratio
    ppl = None
    if include_perplexity:
        ppl = compute_perplexity_ratio(
            orig_success_posts,
            fuzzed_success_posts,
            alpha=0.05,
            max_tokens=512,
        )

    # Mauve
    mauve_score = None
    if include_mauve:
        mauve_score = compute_mauve(
            orig_success_posts,
            fuzzed_success_posts,
            lang=lang,
        )

    return {
        "attack_succ_rate": round(float(attack_succ_rate), 4),
        "iters": iter_stats,
        "bertscore": bertscore,
        "perplexity": ppl,
        "mauve": mauve_score,
    }
