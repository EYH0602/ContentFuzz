import pandas as pd
import fire
from returns.result import Success, Failure

from contentfuzz.cls import StanceAnalyzer, OpenAIAnalyzer, COLA

PRINT_FORMAT = """
Text: {text}
Stance: {result.label}
Rationale: {result.rationale}
Confidence: {prob}
"""


def main(input_file_path: str) -> None:
    """Main entry point to run ContentFuzz"""

    dataset = pd.read_csv(input_file_path)

    # analyzer: StanceAnalyzer = OpenAIAnalyzer()
    analyzer: StanceAnalyzer = COLA()
    for _, row in dataset.iterrows():
        text = row["Text"]
        match analyzer.analyze(text):
            case Success((result, prob)):
                print(PRINT_FORMAT.format(text=text, result=result, prob=prob))
            case Failure(exception):
                print(f"Error analyzing text: {exception}")

        break


if __name__ == "__main__":
    fire.Fire(main)
