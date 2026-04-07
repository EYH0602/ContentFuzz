import argparse
import logging
import os
import random
import sys
from typing import get_args

import orjson
from orjsonl import orjsonl
from returns.result import Failure, Success
from tqdm import tqdm

from contentfuzz.cls import COLA, Analyzer, Encoder, StanceAnalyzer, ZeroshotAnalyzer
from contentfuzz.evaluate import (
    compute_fuzz_metrics,
    compute_metrics,
    get_correct_tasks,
    load_gen_results,
    print_eval_metrics,
)
from contentfuzz.fuzz import Fuzzer, Mutator
from contentfuzz.fuzz.seed_scheduler import (
    FIFOScheduler,
    PriorityRandomScheduler,
    PriorityScheduler,
    RandomScheduler,
    SchedulerChoice,
    SeedScheduler,
)
from contentfuzz.stance_dataset import (
    Dataset,
    DatasetLangMap,
    StanceDataEntry,
    StanceDataset,
    load_c_stance,
    load_sem16,
    load_vast,
)
from contentfuzz.utils import SEED, get_default_atk_output_path, get_skip_cnt


def get_fuzzer_state_path(attack_output_path: str) -> str:
    """Return the JSON file path that stores temperature scheduling statistics.

    We keep the base of the attack output file and append `.fuzzer_stat.json` so
    users can correlate scheduler state with the corresponding fuzz results.
    """
    base, ext = os.path.splitext(attack_output_path)
    if ext == ".jsonl":
        return f"{base}.fuzzer_stat.json"
    return f"{attack_output_path}.fuzzer_stat.json"


def main(  # pylint: disable=too-many-locals, too-many-arguments, R0917, R0915
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
    n_iters: int = 300,
) -> None:
    """Run the mutation-based fuzzing loop on correctly classified tasks.

    Args:
        dataset_name: Dataset name matching the baseline classification run.
        analyzer_name: Analyzer used to score mutated posts (`zeroshot`, `cola`,
            or `encoder`). The analyzer model defaults to `cls_model`.
        cls_output_path: JSONL file with baseline classifier outputs.
        fuzzer_model: LLM used by the `Mutator` to rewrite text.
        cls_model: LLM used when re-scoring fuzzed candidates.
        attack_output_path: Where to write fuzz results (auto-derived if None).
        temperature: Optional fixed sampling temperature; when omitted the
            mutator enables adaptive temperature scheduling.
        mutate_n: Number of rewrites to request per iteration.
        sample_n: Optional number of correctly classified tasks to fuzz.
        schedule: Seed scheduler to prioritize candidates.
        n_iters: Maximum fuzzing iterations per task before giving up.

    The function restores/respects partially written outputs, streams new fuzzed
    samples to JSONL, prints evaluation metrics, and records scheduler statistics
    inside a `.fuzzer_stat.json` companion file.
    """

    if attack_output_path is None:
        output_dir = os.path.abspath("fuzz")
        os.makedirs(output_dir, exist_ok=True)
        temp_ext = "+temp-sched" if temperature is None else f"+temp-{str(temperature)}"
        sched_ext = f"+{schedule}"
        iter_ext = f"+iters-{n_iters}"
        attack_output_path = get_default_atk_output_path(
            output_dir,
            cls_output_path,
            f"{fuzzer_model}{temp_ext}{sched_ext}{iter_ext}",
        )

    random.seed(SEED)
    dataset: StanceDataset
    match dataset_name:
        case "c-stance-a" | "c-stance-b":
            dataset = load_c_stance(dataset_name, "test")
        case "sem16":
            dataset = load_sem16("test")
        case "vast":
            dataset = load_vast("test")
    lang = DatasetLangMap[dataset_name]

    analyzer: StanceAnalyzer
    match analyzer_name:
        case "zeroshot":
            analyzer = ZeroshotAnalyzer(model=cls_model)
        case "encoder":
            analyzer = Encoder(model=cls_model)
        case "cola":
            analyzer = COLA(model=cls_model, language=lang)

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

    mutator = Mutator(
        model=fuzzer_model,
        n=mutate_n,
        temperature=temperature,
        lang=lang,
    )
    fuzzer = Fuzzer(analyzer, mutator, scheduler)

    temp_msg = (
        "+ temperature scheduling"
        if temperature is None
        else f"+ temperature = {temperature}"
    )
    logging.info(f"Fuzzing with {fuzzer_model} {temp_msg}")

    if sample_n is not None:
        logging.info(f"Sampling {sample_n} from total {len(ct)} fuzzing tasks")
        if sample_n <= 0:
            logging.error(f"sample_n must be positive, got {sample_n}")
            sys.exit(1)
        ct = random.sample(ct, k=sample_n)

    # skip existing results
    skip_count = get_skip_cnt(attack_output_path)
    ct = ct[skip_count:]
    if skip_count > 0:
        logging.info(
            f"Found {skip_count} results in {attack_output_path}, skipping them..."
        )

    total_tasks_to_fuzz = len(ct)

    for t in tqdm(ct, total=total_tasks_to_fuzz):
        task: StanceDataEntry = t  # type: ignore
        match fuzzer.runs(task, n_iters=n_iters):
            case Success(((mutated_text, stance, confidence), iter_cnt)):
                log_obj = task | {
                    "new_text": mutated_text,
                    "predicted": stance,
                    "confidence": confidence,
                    "iteration": iter_cnt,
                }
                orjsonl.append(attack_output_path, log_obj)
            case Failure(err):
                err_obj = task | {"error": err}
                orjsonl.append(attack_output_path, err_obj)

    df = load_gen_results(attack_output_path)
    fuzz_metrics = compute_fuzz_metrics(df, lang=lang)
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
        help="Path to baseline generation results JSONL produced by run_cls.py.",
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
    parser.add_argument(
        "-ni",
        "--n-iters",
        dest="n_iters",
        type=int,
        default=300,
        help="Number of iterations to run the fuzzer.",
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
        n_iters=args.n_iters,
    )
