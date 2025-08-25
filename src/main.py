import os

import pandas as pd
import fire

from contentfuzz.cls import OpenAIAnalyzer


def main(input_file_path: str) -> None:
    """Main entry point to run ContentFuzz"""

    dataset = pd.read_csv(input_file_path)

    analyzer = OpenAIAnalyzer()
    for _, row in dataset.iterrows():
        text = row["Text"]
        result, prob = analyzer.analyze(text)
        print(
            f"Text: {text}\nStance: {result.label}\nRationale: {result.rationale}\nConfidence: {prob}\n"
        )


if __name__ == "__main__":
    fire.Fire(main)
