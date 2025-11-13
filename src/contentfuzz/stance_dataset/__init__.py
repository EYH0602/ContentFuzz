from typing import Literal
from .utils import negate_stance
from .c_stance import load_c_stance, CSTANCEChoices
from .sem16 import load_sem16
from ._base import StanceDataEntry, StanceDataset
from ..utils import Language

Dataset = Literal[CSTANCEChoices, "sem16"]  # todo: add more datasets
DatasetLangMap: dict[Dataset, Language] = {
    "c-stance-a": "zh",
    "c-stance-b": "zh",
    "sem16": "en",
}

__all__ = [
    "negate_stance",
    "load_c_stance",
    "load_sem16",
    "CSTANCEChoices",
    "Dataset",
    "StanceDataEntry",
    "StanceDataset",
    "DatasetLangMap",
]
