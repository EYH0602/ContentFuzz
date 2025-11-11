import argparse
import os
from typing import get_args
import random
import logging
import sys

from returns.result import Success, Failure
from orjsonl import orjsonl
from tqdm import tqdm
import orjson
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
    compute_metrics,
    print_eval_metrics,
    compute_fuzz_metrics,
)
from contentfuzz.stance_dataset import (
    load_c_stance,
    load_sem16,
    StanceDataEntry,
    Dataset,
    StanceDataset,
)
from contentfuzz.fuzz import Mutator, Fuzzer
from contentfuzz.fuzz.seed_scheduler import (
    SchedulerChoice,
    RandomScheduler,
    PriorityScheduler,
    FIFOScheduler,
    PriorityRandomScheduler,
    SeedScheduler,
)
from contentfuzz.cls import ZeroshotAnalyzer, Analyzer, StanceAnalyzer, Encoder, COLA
from contentfuzz.utils import get_default_atk_output_path, SEED


def get_skip_cnt(file_path: str) -> int:
    """count number of a record JSONL file"""
    if not os.path.isfile(file_path):
        return 0
    with open(file_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def get_fuzzer_state_path(attack_output_path: str) -> str:
    """Derive the path for storing serialized fuzzer state."""
    base, ext = os.path.splitext(attack_output_path)
    if ext == ".jsonl":
        return f"{base}.fuzzer_stat.json"
    return f"{attack_output_path}.fuzzer_stat.json"


def main(  # pylint: disable=too-many-locals, too-many-arguments, R0917
    dataset_name: Dataset,
    analyzer_name: Analyzer,
    cls_output_path: str,
    fuzzer_model: str = "gemini-2.5-flash-lite",
    cls_model: str = "gemini-2.5-flash-lite",
    attack_output_path: str | None = None,
    temperature: float | None = None,
    mutate_n: int = 5,
    sample_n: int | None = None,
    schedule: SchedulerChoice = "priority",
) -> None:
    """Main entry point to run fuzzing in ContentFuzz

    Usage:
    python src/run_fuzz.py -h
    """

    if attack_output_path is None:
        output_dir = os.path.abspath("fuzz")
        os.makedirs(output_dir, exist_ok=True)
        temp_ext = "" if temperature is None else f"+temp-{str(temperature)}"
        sched_ext = f"+{schedule}"
        attack_output_path = get_default_atk_output_path(
            output_dir,
            cls_output_path,
            f"{fuzzer_model}{temp_ext}{sched_ext}",
        )

    random.seed(SEED)
    dataset: StanceDataset
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            dataset = load_c_stance(dataset_name, "test")
        case "sem16":
            dataset = load_sem16("test")

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=cls_model)
        case "encoder":
            analyzer = Encoder(model=cls_model)
        case "cola":
            analyzer = COLA(model=cls_model)

    scheduler: SeedScheduler
    match schedule:
        case "fifo":
            scheduler = FIFOScheduler()
        case "priority":
            scheduler = PriorityScheduler()
        case "random":
            scheduler = RandomScheduler()
        case "priority_random":
            scheduler = PriorityRandomScheduler()

    gen_results = load_gen_results(cls_output_path)

    assert len(dataset) == len(gen_results), "Dataset and results length mismatch"

    logging.info(f"{analyzer_name} w/ {cls_model} on {dataset_name}")
    metrics = compute_metrics(gen_results)
    print_eval_metrics(metrics)

    correct_tasks, _ = get_correct_tasks(dataset, gen_results)
    ct = correct_tasks.to_dict("records")

    mutator = Mutator(model=fuzzer_model, n=mutate_n, temperature=temperature)
    fuzzer = Fuzzer(analyzer, mutator, scheduler)

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
        logging.info(
            f"Found {skip_count} results in {attack_output_path}, skipping them..."
        )

    if sample_n is not None:
        logging.info(f"Sampling {sample_n} from total {len(ct)} fuzzing tasks")
        if sample_n <= 0:
            logging.error(f"sample_n must be positive, got {sample_n}")
            sys.exit(1)
        ct = random.sample(ct, k=sample_n)

    total_tasks_to_fuzz = len(ct)

    for t in tqdm(ct, total=total_tasks_to_fuzz):
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

    df = load_gen_results(attack_output_path)
    fuzz_metrics = compute_fuzz_metrics(df)
    print_eval_metrics(fuzz_metrics)

    assert attack_output_path is not None
    fuzzer_state_path = get_fuzzer_state_path(attack_output_path)
    temp_counts = mutator.get_temperature_stats()
    fuzzer_stats = {
        "temperature_scheduling": temp_counts,
    }
    with open(fuzzer_state_path, "wb") as state_file:
        state_file.write(
            orjson.dumps(
                fuzzer_stats,
                option=(
                    orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS
                ),
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run ContentFuzz mutation attacks against classification results."
    )
    dataset_choices = get_args(Dataset)
    analyzer_choices = get_args(Analyzer)
    schedule_choices = get_args(SchedulerChoice)
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
        "-sn",
        "--sample-n",
        dest="sample_n",
        type=int,
        default=None,
        help="Optional number of tasks to sample before fuzzing.",
    )
    parser.add_argument(
        "-mn",
        "--mutate-n",
        dest="mutate_n",
        type=int,
        default=5,
        help="Number of mutations to generate per task.",
    )
    parser.add_argument(
        "-s",
        "--schedule",
        dest="schedule",
        choices=schedule_choices,
        default="priority",
        help="Seed scheduling strategy.",
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
        sample_n=args.sample_n,
        schedule=args.schedule,
    )
