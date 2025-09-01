import logging

import fire
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
)
from contentfuzz.datasets import load_c_stance, StanceDataEntry
from contentfuzz.fuzzer import Mutator
from contentfuzz.cls import OpenAIAnalyzer


def main(
    dataset_file_path: str = "C-STANCE/c_stance_dataset/subtaskA/raw_test_all_onecol.csv",
    generate_result_path: str = "results/c_stance_A_gpt-4.1-nano.jsonl",
) -> None:
    """Main entry point to run ContentFuzz"""

    dataset = load_c_stance(dataset_file_path)
    gen_results = load_gen_results(generate_result_path)

    assert len(dataset) == len(gen_results), "Dataset and results length mismatch"

    correct_tasks, correct_results = get_correct_tasks(dataset, gen_results)
    # ct = correct_tasks.iloc[0].to_dict()
    ct = correct_tasks.to_dict("records")

    task: StanceDataEntry = ct[0]  # type: ignore
    print("Original Task:")
    print(task)
    mutator = Mutator()
    classifier = OpenAIAnalyzer()

    original_response = correct_results.loc[0].to_dict()
    print("Original Response:")
    print(original_response)

    print("Trying to steer")
    mutated = mutator.steer(task)
    print(mutated)
    r = classifier.analyze(mutated, target=task["target"])
    print(r)

    print("Trying to rewrite")
    mutated = mutator.rewrite(task)
    print(mutated)
    r = classifier.analyze(mutated, target=task["target"])
    print(r)

    print("Trying to generate TL;DR")
    mutated = mutator.tldr(task)
    print(mutated)
    r = classifier.analyze(mutated, target=task["target"])
    print(r)

    print("Trying to generate Hash-Tags")
    mutated = mutator.tags(task)
    print(mutated)
    r = classifier.analyze(mutated, target=task["target"])
    print(r)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
