from .utils import negate_stance
from .c_stance import load_c_stance, CSTANCEChoices
from ._base import StanceDataEntry, StanceDataset

Dataset = CSTANCEChoices  # todo: add more datasets

__all__ = [
    "negate_stance",
    "load_c_stance",
    "CSTANCEChoices",
    "Dataset",
    "StanceDataEntry",
    "StanceDataset",
]
