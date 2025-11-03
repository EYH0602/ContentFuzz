import argparse
import os
from typing import get_args
import random

from returns.result import Success, Failure
from orjsonl import orjsonl
from tqdm import tqdm
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
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


def main(  # pylint: disable=too-many-locals, too-many-arguments, R0917
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    cls_output_path: str,
    model: str = "gemini-2.5-flash-lite",
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
            f"{model}{temp_ext}",
        )

    random.seed(SEED)
    dataset: StanceDataset
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            dataset = load_c_stance(dataset_name, "test")

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=model)
        case "encoder":
            analyzer = Encoder(model=model)

    gen_results = load_gen_results(cls_output_path)

    assert len(dataset) == len(gen_results), "Dataset and results length mismatch"

    correct_tasks, _ = get_correct_tasks(dataset, gen_results)
    # ct = correct_tasks.iloc[0].to_dict()
    ct = correct_tasks.to_dict("records")

    mutator = Mutator(model=model, n=mutate_n, temperature=temperature)
    fuzzer = Fuzzer(analyzer, mutator)

    for t in tqdm(ct):
        task: StanceDataEntry = t  # type: ignore
        match fuzzer.runs(task):
            case Success((mutated_text, stance, confidence)):
                log_obj = task | {
                    "new_text": mutated_text,
                    "predicted": stance,
                    "confidence": confidence,
                }
                orjsonl.append(attack_output_path, log_obj)
            case Failure(_):
                continue
    n_succ = len(orjsonl.load(attack_output_path))
    n_total = len(ct)
    print(f"Attack success rate: {n_succ}/{n_total} = {n_succ/n_total:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ContentFuzz mutation attacks against classification results."
    )
    dataset_choices = get_args(Dataset)
    parser.add_argument(
        "dataset_name",
        choices=dataset_choices,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "-co",
        "--cls-output-path",
        dest="cls_output_path",
        default="results/c_stance_A_gpt-4.1-nano.jsonl",
        help=(
            "Path to baseline generation results JSONL; defaults to "
            "results/c_stance_A_gpt-4.1-nano.jsonl."
        ),
    )
    parser.add_argument(
        "-ao",
        "--attack-output-path",
        dest="attack_output_path",
        default="results/c_stance_A_gpt-4.1-nano.attack.jsonl",
        help=(
            "Path to write attack results JSONL; defaults to "
            "results/c_stance_A_gpt-4.1-nano.attack.jsonl."
        ),
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        cls_output_path=args.cls_output_path,
        attack_output_path=args.attack_output_path,
    )
