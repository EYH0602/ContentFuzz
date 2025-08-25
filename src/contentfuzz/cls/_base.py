from typing import Literal
from pydantic import BaseModel

Label = Literal["Favor", "Against", "Neutral"]


class StanceOutput(BaseModel):
    """Classifier response format"""

    label: Label
    rationale: str


AnalysisOutput = tuple[StanceOutput, float | None]
