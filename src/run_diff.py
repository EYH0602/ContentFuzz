import fire
from contentfuzz.evaluate import (
    load_gen_results,
    get_correct_tasks,
)
from contentfuzz.datasets import load_c_stance, StanceDataEntry
from contentfuzz.fuzz import Mutator
from contentfuzz.cls import OpenAIAnalyzer


def mutate_and_classify(mutator, classifier, task: StanceDataEntry):
    """Mutate the task using the mutator and classify using the classifier"""
    mutated = mutator.mutate(task)
    for i, m in enumerate(mutated):
        r = classifier.analyze(m, target=task["target"])
        print(f"Variant {i}: {m}")
        print(f"Classification: {r}")
    return mutated


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
    for m in mutator.mutators:
        print(f"Using mutator: {m.__name__}")
        mutate_and_classify(m, classifier, task)


if __name__ == "__main__":
    fire.Fire(main)
