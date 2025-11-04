import argparse
import os
from typing import get_args
import random
import logging

from returns.result import Success, Failure
from orjsonl import orjsonl
from tqdm import tqdm
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
    compute_metrics,
    print_eval_metrics,
)
from contentfuzz.stance_dataset import (
    load_c_stance,
    StanceDataEntry,
    Dataset,
    StanceDataset,
)
from contentfuzz.fuzz import Mutator, Fuzzer
from contentfuzz.cls import ZeroshotAnalyzer, Analyzer, StanceAnalyzer, Encoder
from contentfuzz.utils import get_default_atk_output_path, SEED


def get_skip_cnt(file_path: str) -> int:
    """count number of a record JSONL file"""
    with open(file_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def main(  # pylint: disable=too-many-locals, too-many-arguments, R0917
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    cls_output_path: str,
    fuzzer_model: str = "gemini-2.5-flash-lite",
    cls_model: str = "gemini-2.5-flash-lite",
    attack_output_path: str | None = None,
    temperature: float | None = None,
    mutate_n: int = 5,
) -> None:
    """Main entry point to run fuzzing in ContentFuzz

    Usage:
    python src/run_fuzz.py -h
    """

    if attack_output_path is None:
        output_dir = os.path.abspath("fuzz")
        os.makedirs(output_dir, exist_ok=True)
        temp_ext = "" if temperature is None else f"-{str(temperature)}"
        attack_output_path = get_default_atk_output_path(
            output_dir,
            cls_output_path,
            f"{fuzzer_model}{temp_ext}",
        )

    random.seed(SEED)
    dataset: StanceDataset
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            dataset = load_c_stance(dataset_name, "test")

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=cls_model)
        case "encoder":
            analyzer = Encoder(model=cls_model)

    gen_results = load_gen_results(cls_output_path)

    assert len(dataset) == len(gen_results), "Dataset and results length mismatch"

    logging.info(f"{analyzer_name} w/ {cls_model} on {dataset_name}")
    metrics = compute_metrics(gen_results)
    print_eval_metrics(metrics)

    correct_tasks, _ = get_correct_tasks(dataset, gen_results)
    ct = correct_tasks.to_dict("records")

    mutator = Mutator(model=fuzzer_model, n=mutate_n, temperature=temperature)
    fuzzer = Fuzzer(analyzer, mutator)

    temp_msg = (
        "+ temperature scheduling"
        if temperature is None
        else f"+ temperature = {temperature}"
    )
    logging.info(f"Fuzzing with {fuzzer_model} {temp_msg}")
    n_succ = 0

    # skip existing results
    skip_count = get_skip_cnt(attack_output_path)
    ct = ct[skip_count:]
    if skip_count > 0:
        logging.info(f"Found {skip_count} results in {attack_output_path}, skipping...")

    for t in tqdm(ct, initial=skip_count, total=len(ct)):
        task: StanceDataEntry = t  # type: ignore
        match fuzzer.runs(task):
            case Success((mutated_text, stance, confidence)):
                log_obj = task | {
                    "new_text": mutated_text,
                    "predicted": stance,
                    "confidence": confidence,
                }
                orjsonl.append(attack_output_path, log_obj)
                n_succ += 1
            case Failure(err):
                err_obj = task | {"error": err}
                orjsonl.append(attack_output_path, err_obj)

    n_total = len(ct)
    print(f"Attack success rate (this run): {n_succ}/{n_total} = {n_succ/n_total:.2%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run ContentFuzz mutation attacks against classification results."
    )
    dataset_choices = get_args(Dataset)
    analyzer_choices = get_args(Analyzer)
    parser.add_argument(
        "dataset_name",
        choices=dataset_choices,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "analyzer_name",
        choices=analyzer_choices,
        help="Analyzer strategy used to score mutated samples.",
    )
    parser.add_argument(
        dest="cls_output_path",
        help=(
            "Path to baseline generation results JSONL; defaults to "
            "results/c_stance_A_gpt-4.1-nano.jsonl."
        ),
    )
    parser.add_argument(
        "-ao",
        "--attack-output-path",
        dest="attack_output_path",
        default=None,
        help=(
            "Path to write attack results JSONL; defaults to the automatic "
            "path derived from the classifier outputs."
        ),
    )
    parser.add_argument(
        "-fm",
        "--fuzzer-model",
        dest="fuzzer_model",
        default="gemini-2.5-flash-lite",
        help="Model to use for fuzzing.",
    )
    parser.add_argument(
        "-cm",
        "--cls-model",
        dest="cls_model",
        default="gemini-2.5-flash-lite",
        help="Model to use for stance analysis.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        type=float,
        default=None,
        help="Optional temperature to sample mutations. If None, enable temperature scheduling",
    )
    parser.add_argument(
        "-n",
        "--mutate-n",
        dest="mutate_n",
        type=int,
        default=5,
        help="Number of mutations to generate per task.",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        analyzer_name=args.analyzer_name,
        cls_output_path=args.cls_output_path,
        fuzzer_model=args.fuzzer_model,
        cls_model=args.cls_model,
        attack_output_path=args.attack_output_path,
        temperature=args.temperature,
        mutate_n=args.mutate_n,
    )
