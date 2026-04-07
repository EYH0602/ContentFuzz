from typing import Literal, TypeGuard, get_args
from .utils import negate_stance
from .c_stance import load_c_stance, CSTANCEChoices
from .sem16 import load_sem16
from .vast import load_vast
from ._base import StanceDataEntry, StanceDataset
from ..utils import Language

Dataset = Literal[CSTANCEChoices, "sem16", "vast"]
DatasetLangMap: dict[Dataset, Language] = {
    "c-stance-a": "zh",
    "c-stance-b": "zh",
    "sem16": "en",
    "vast": "en",
}


def is_dataset(dataset: str) -> TypeGuard[Dataset]:
    """check if the dataset string is a valid Dataset"""
    return dataset in get_args(Dataset)


__all__ = [
    "negate_stance",
    "load_c_stance",
    "load_sem16",
    "load_vast",
    "CSTANCEChoices",
    "Dataset",
    "StanceDataEntry",
    "StanceDataset",
    "DatasetLangMap",
    "is_dataset",
]
