from .utils import translate_stance, negate_stance
from .c_stance import load_c_stance, C_STANCE
from ._base import StanceDataEntry, StanceDataset

Dataset = C_STANCE  # todo: add more datasets

__all__ = [
    "translate_stance",
    "negate_stance",
    "load_c_stance",
    "C_STANCE",
    "Dataset",
    "StanceDataEntry",
    "StanceDataset",
]
