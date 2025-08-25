from typing import Literal, Protocol, runtime_checkable
from pydantic import BaseModel

from returns.result import ResultE


Label = Literal["Favor", "Against", "Neutral"]


class StanceOutput(BaseModel):
    """Classifier response format"""

    label: Label
    rationale: str


AnalysisOutput = tuple[StanceOutput, float | None]


@runtime_checkable
class StanceAnalyzer(Protocol):
    """Base protocol for all analyzers"""

    def analyze(self, text: str, target: str | None = None) -> ResultE[AnalysisOutput]:
        """
        Analyze the given text and return the analysis output.

        Args:
            text (str): The text to analyze.
            target (str | None): The target entity to analyze the text against.
                Some text may not be assigned a target, so the default is None.
                It is the specific Analyzer's responsibility to handle it appropriately.
        """
        ...
