import logging
from typing import Literal

from ._base import StanceAnalyzer, AnalysisOutput, StanceOutput
from .zeroshot import OpenAIAnalyzer
from .cola import COLA

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

ANALYZER = Literal["zero-shot", "cola"]

__all__ = [
    "StanceAnalyzer",
    "AnalysisOutput",
    "StanceOutput",
    "OpenAIAnalyzer",
    "COLA",
    "ANALYZER",
]
