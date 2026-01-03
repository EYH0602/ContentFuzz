import argparse
import logging
import os
import random
from typing import get_args

import pandas as pd

from contentfuzz.cls import (
    COLA,
    Analyzer,
    Encoder,
    StanceAnalyzer,
    ZeroshotAnalyzer,
)
from contentfuzz.cross_model import compute_cross_model_esr, load_cross_model_dataset
from contentfuzz.evaluate import (
    load_gen_results,
)
from contentfuzz.run import run_batch_generation
from contentfuzz.stance_dataset import (
    Dataset,
    DatasetLangMap,
    StanceDataset,
)
from contentfuzz.utils import SEED, get_skip_cnt


def get_default_cross_model_output_path(
    output_dir: str, analyzer_name: str, model: str, file_name: str
) -> str:
    """Construct default cross model output path.
    We replace '/' in model names with '--' to avoid issues with file paths.
    Naming format:
        crossed_analyzer+model=input_file_name.jsonl
    """
    sanitized_model_name = model.replace("/", "--")
    output_result_path = os.path.join(
        output_dir,
        f"{analyzer_name}+{sanitized_model_name}={file_name}",
    )
    return output_result_path


def main(
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    input_result_path: str,
    model: str = "gemini-2.5-flash-lite",
    output_result_path: str | None = None,
    batch_size: int = 1,
) -> None:
    """Re-score successful fuzzed posts with a different analyzer/model combo.

    Args:
        dataset_name: Dataset identifier that matches the fuzzed tasks.
        analyzer_name: Analyzer literal (`zeroshot`, `cola`, or `encoder`).
        input_result_path: JSONL output from a previous fuzzing run. Only rows
            that flipped the original classifier are replayed.
        model: LLM identifier to pass to the analyzer.
        output_result_path: Optional JSONL file where new predictions are saved.
        batch_size: Number of tasks to send to the analyzer concurrently.

    The function reads fuzzing outputs, extracts the altered text, skips rows
    that were already scored (based on the output JSONL), runs the requested
    analyzer, and prints the resulting cross-model escape success rate (ESR).
    """

    if output_result_path is None:
        output_dir = os.path.abspath(f"cross_model/{dataset_name}")
        os.makedirs(output_dir, exist_ok=True)

        file_base = os.path.basename(input_result_path)
        output_result_path = get_default_cross_model_output_path(
            output_dir, analyzer_name, model, file_base
        )
    random.seed(SEED)
    lang = DatasetLangMap[dataset_name]

    gen_results = load_gen_results(input_result_path)
    success_tasks: StanceDataset = load_cross_model_dataset(gen_results)

    skip_count = get_skip_cnt(output_result_path)
    if skip_count > 0:
        logging.info(
            f"Found {skip_count} results in {output_result_path}, skipping them..."
        )
        success_tasks = success_tasks[skip_count:]

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=model)
        case "cola":
            analyzer = COLA(model=model, language=lang)
        case "encoder":
            analyzer = Encoder(model=model)

    logging.info(f"Running {analyzer.__class__.__name__} with {analyzer.model}.")
    results = run_batch_generation(
        success_tasks,
        analyzer,
        batch_size=batch_size,
        output_result_path=output_result_path,
    )

    cross_model_esr = compute_cross_model_esr(pd.DataFrame(results))
    print(f"Cross-Model ESR: {cross_model_esr}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_choices = get_args(Dataset)
    analyzer_choices = get_args(Analyzer)
    parser = argparse.ArgumentParser(
        description="Run ContentFuzz stance classification experiment."
    )
    parser.add_argument(
        "dataset_name",
        choices=dataset_choices,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "analyzer_name",
        choices=analyzer_choices,
        help="Analyzer to run.",
    )
    parser.add_argument(
        "input_result_path",
        help="Path to input generation results JSONL.",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-2.5-flash-lite",
        help="Gemini Model to use for generation.",
    )
    parser.add_argument(
        "-o",
        "--output-result-path",
        dest="output_result_path",
        help=(
            "Optional path to store generation results JSONL; defaults to "
            "cross_model/{dataset_name}/{analyzer}+{model}={input_file_name}."
        ),
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
        help="If set, run generation in batches of the given size. Defaults to 1.",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        analyzer_name=args.analyzer_name,
        input_result_path=args.input_result_path,
        model=args.model,
        output_result_path=args.output_result_path,
        batch_size=args.batch_size,
    )
