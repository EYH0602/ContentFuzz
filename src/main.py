import logging

import pandas as pd
import fire
import orjson
from returns.result import Success, Failure
from tqdm import tqdm

from contentfuzz.cls import StanceAnalyzer, OpenAIAnalyzer, COLA
from contentfuzz.datasets import translate_stance

PRINT_FORMAT = """
Text: {text}
Stance: {result.label}
Rationale: {result.rationale}
Confidence: {prob}
"""


def main(input_file_path: str, output_result_path: str = "out.jsonl") -> None:
    """Main entry point to run ContentFuzz"""

    dataset = pd.read_csv(input_file_path)

    analyzer: StanceAnalyzer = OpenAIAnalyzer()
    logging.info(f"Running analysis with {analyzer.__class__.__name__}.")
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        text = row["Text"]
        match analyzer.analyze(text):
            case Success((result, prob)):
                log = {
                    "truth": translate_stance(row["Stance 1"]),
                    "predicted": result.label,
                    "rationale": result.rationale,
                    "confidence": prob,
                }
                with open(output_result_path, "ab") as f:
                    f.write(orjson.dumps(log, option=orjson.OPT_APPEND_NEWLINE))
            case Failure(exception):
                print(f"Error analyzing text: {exception}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
