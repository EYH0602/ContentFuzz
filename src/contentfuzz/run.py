import logging

from deprecated import deprecated
from orjsonl import orjsonl
from returns.result import Failure, Success

from .cls import StanceAnalyzer
from .evaluate import GenResult
from .stance_dataset import StanceDataset


@deprecated("Use `run_batch_generation` with `batch_size=1` instead.")
def run_generation(
    dataset: StanceDataset,
    analyzer: StanceAnalyzer,
    output_result_path: str | None = None,
) -> list[GenResult]:
    """Legacy wrapper calling `run_batch_generation` with `batch_size=1`."""
    return run_batch_generation(
        dataset,
        analyzer,
        output_result_path=output_result_path,
        batch_size=1,
    )


def run_batch_generation(
    dataset: StanceDataset,
    analyzer: StanceAnalyzer,
    output_result_path: str | None = None,
    batch_size: int | None = None,
) -> list[GenResult]:
    """run generation experiments

    Args:
        dataset (StanceDataset): dataset with columns ["text", "stance", "target"]
        analyzer (StanceAnalyzer): the classification analyzer to use
        output_result_path (str | None, optional): path to the output result file (JSONL).
            Defaults to None.
            If None, the file will be named `out_<analyzer_class_name>.jsonl`
        batch_size (int, optional): Number of samples to process together. Defaults to 8.

    Returns:
        list[GenResult]: List of generation results
    """

    if output_result_path is None:
        output_result_path = f"out_{analyzer.__class__.__name__}.jsonl"

    results: list[GenResult] = []
    tasks = [(row["text"], row["target"]) for row in dataset]
    cls_results = analyzer.analyze(tasks, batch_size=batch_size)
    for row, result in zip(dataset, cls_results):
        log_obj: GenResult = {
            "truth": row["stance"],
            "predicted": "error",  # in case analyze failed
            "confidence": 0.0,
        }
        match result:
            case Success((result, prob)):
                log_obj["predicted"] = result
                log_obj["confidence"] = prob
            case Failure(_ as e):
                logging.error(f"Error analyzing text: {e}")

        results.append(log_obj)
        orjsonl.append(output_result_path, log_obj)
    return results
