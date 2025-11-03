import fire
from returns.maybe import Maybe, Some, Nothing  # noqa: F401
from orjsonl import orjsonl
from tqdm import tqdm
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
)
from contentfuzz.stance_dataset import load_c_stance, StanceDataEntry, C_STANCE
from contentfuzz.fuzz import Mutator, Fuzzer
from contentfuzz.cls import ZeroshotAnalyzer


def mutate_and_classify(mutate, classifier, task: StanceDataEntry):
    """Mutate the task using the mutator and classify using the classifier"""
    mutated = mutate(task)
    for i, m in enumerate(mutated):
        r = classifier.analyze(m, target=task["target"])
        print(f"Variant {i}: {m}")
        print(f"Classification: {r}")
    return mutated


def main(
    dataset_name: C_STANCE,
    generate_result_path: str = "results/c_stance_A_gpt-4.1-nano.jsonl",
    attack_result_path: str = "results/c_stance_A_gpt-4.1-nano.attack.jsonl",
) -> None:
    """Main entry point to run ContentFuzz"""

    dataset = load_c_stance(dataset_name, "test")
    gen_results = load_gen_results(generate_result_path)

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
            case Some((mutated_text, stance, confidence)):
                log_obj = task | {
                    "new_text": mutated_text,
                    "predicted": stance.label,
                    "rationale": stance.rationale,
                    "confidence": confidence,
                }
                orjsonl.append(attack_result_path, log_obj)
            case Nothing:  # noqa: F811, F841
                continue
    n_succ = len(orjsonl.load(attack_result_path))
    n_total = len(ct)
    print(f"Attack success rate: {n_succ}/{n_total} = {n_succ/n_total:.2%}")


if __name__ == "__main__":
    fire.Fire(main)
