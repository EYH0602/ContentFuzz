from typing import Protocol, runtime_checkable, Literal

from pydantic import BaseModel
from returns.result import ResultE


StanceOutput = Literal["Favor", "Against", "Neutral"]
AnalysisOutput = tuple[StanceOutput, float | None]


class ClassifierOutput(BaseModel):
    """for OpenAI's structured output"""

    stance: StanceOutput


@runtime_checkable
class StanceAnalyzer(Protocol):
    """Base protocol for all analyzers"""

    model: str

    def analyze(self, text: str, target: str) -> ResultE[AnalysisOutput]:
        """
        Analyze the given text and return the analysis output.

        Args:
            text (str): The text to analyze.
            target (str): The target entity to analyze the text against.
        """
        ...
