from typing import Protocol, runtime_checkable
from pydantic import BaseModel


class AnalysisOutput(BaseModel):
    """Base class for all analysis outputs"""

    stance: str
    rationale: str
    prob: float | None
