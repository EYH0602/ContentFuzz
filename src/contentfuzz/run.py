import logging

from tqdm import tqdm
from returns.result import Success, Failure
from orjsonl import orjsonl

from .cls import StanceAnalyzer
from .evaluate import GenResult
from .stance_dataset import StanceDataset


def run_generation(
    dataset: StanceDataset,
    analyzer: StanceAnalyzer,
    output_result_path: str | None = None,
) -> list[GenResult]:
    """run generation experiments

    Args:
        dataset (StanceDataset): dataset with columns ["text", "stance", "target"]
        analyzer (StanceAnalyzer): the classification analyzer to use
        output_result_path (str | None, optional): path to the output result file (JSONL).
            Defaults to None.
            If None, the file will be named `out_<analyzer_class_name>.jsonl`

    Returns:
        list[GenResult]: List of generation results
    """

    if output_result_path is None:
        output_result_path = f"out_{analyzer.__class__.__name__}.jsonl"

    results: list[GenResult] = []

    for row in tqdm(dataset):
        text = row["text"]
        target = row["target"]
        log_obj: GenResult = {
            "truth": row["stance"],
            "predicted": "error",  # in case analyze failed
            "confidence": 0.0,
        }
        match analyzer.analyze(text, target=target):
            case Success((result, prob)):
                log_obj["predicted"] = result
                log_obj["confidence"] = prob
            case Failure(_ as e):
                logging.error(f"Error analyzing text: {e}")

        results.append(log_obj)
        orjsonl.append(output_result_path, log_obj)
    return results
