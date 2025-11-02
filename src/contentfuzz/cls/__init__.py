import logging
from typing import Literal

from ._base import StanceAnalyzer, AnalysisOutput
from .zeroshot import OpenAIAnalyzer
from .cola import COLA
from .encoder import FinetunedEncoder

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

ANALYZER = Literal["zero-shot", "cola", "encoder"]

__all__ = [
    "StanceAnalyzer",
    "AnalysisOutput",
    "OpenAIAnalyzer",
    "COLA",
    "FinetunedEncoder",
    "ANALYZER",
]
