from typing import Protocol, runtime_checkable

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

    def analyze(
        self, tasks: list[tuple[str, str]], batch_size: int | None = None
    ) -> list[ResultE[AnalysisOutput]]:
        """
        Analyze multiple `(text, target)` pairs in batches and return results in order.

        Args:
            tasks (list[tuple[str, str]]): List of (text, target) pairs.
            batch_size (int | None): Number of pairs to process together. Defaults to None.
                If None, processes all samples concurrently, and let retry handle rate limits.
        """
        ...
