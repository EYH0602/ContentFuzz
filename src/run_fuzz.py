import argparse
from typing import get_args
from returns.result import Success, Failure
from orjsonl import orjsonl
from tqdm import tqdm
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
)
from contentfuzz.stance_dataset import load_c_stance, StanceDataEntry, Dataset
from contentfuzz.fuzz import Mutator, Fuzzer
from contentfuzz.cls import ZeroshotAnalyzer


def main(  # pylint: disable=too-many-locals
    dataset_name: Dataset,
    cls_output_path: str = "results/c_stance_A_gpt-4.1-nano.jsonl",
    attack_output_path: str = "results/c_stance_A_gpt-4.1-nano.attack.jsonl",
) -> None:
    """Main entry point to run fuzzing in ContentFuzz
    
    Usage:
    python src/run_fuzz.py -h
    """

    dataset = load_c_stance(dataset_name, "test")
    gen_results = load_gen_results(cls_output_path)

    assert len(dataset) == len(gen_results), "Dataset and results length mismatch"

    correct_tasks, _ = get_correct_tasks(dataset, gen_results)
    # ct = correct_tasks.iloc[0].to_dict()
    ct = correct_tasks.to_dict("records")

    mutator = Mutator()
    classifier = ZeroshotAnalyzer()
    fuzzer = Fuzzer(classifier, mutator)

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
