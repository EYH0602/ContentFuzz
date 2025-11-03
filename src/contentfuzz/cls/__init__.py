import logging
from typing import Literal

from ._base import StanceAnalyzer, AnalysisOutput
from .zeroshot import ZeroshotAnalyzer
from .cola import COLA
from .encoder import Encoder

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

Analyzer = Literal["zero-shot", "cola", "encoder"]

__all__ = [
    "StanceAnalyzer",
    "AnalysisOutput",
    "ZeroshotAnalyzer",
    "COLA",
    "Encoder",
    "Analyzer",
]
