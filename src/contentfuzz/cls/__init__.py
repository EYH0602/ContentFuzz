import logging
from typing import Literal

from ._base import StanceAnalyzer, AnalysisOutput
from .zeroshot import OpenAIAnalyzer
from .cola import COLA

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

ANALYZER = Literal["zero-shot", "cola"]

__all__ = [
    "StanceAnalyzer",
    "AnalysisOutput",
    "OpenAIAnalyzer",
    "COLA",
    "ANALYZER",
]
