from typing import Protocol, runtime_checkable

from deprecated import deprecated
from pydantic import BaseModel
from returns.result import ResultE

from .._types import Stance

AnalysisOutput = tuple[Stance, float | None]


class ClassifierOutput(BaseModel):
    """for OpenAI's structured output"""

    stance: Stance


@runtime_checkable
class StanceAnalyzer(Protocol):
    """Base protocol for all analyzers"""

    model: str

    @deprecated("Use `batched_analysis` with batch_size=1 instead")
    def analyze(self, text: str, target: str) -> ResultE[AnalysisOutput]:
        """
        Analyze the given text and return the analysis output.

        Args:
            text (str): The text to analyze.
            target (str): The target entity to analyze the text against.
        """
        ...

    def batched_analysis(
        self, tasks: list[tuple[str, str]], batch_size: int | None = None
    ) -> list[ResultE[AnalysisOutput]]:
        """
        Analyze multiple `(text, target)` pairs in batches and return results in order.

        Args:
            entries (list[tuple[str, str]]): List of (text, target) pairs.
            batch_size (int | None): Number of pairs to process together. Defaults to None.
                If None, processes all samples concurrently, and let retry handle rate limits.
        """
        ...
