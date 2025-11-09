from typing import Any, Mapping, cast

from datasets import load_dataset

from ._base import SPLITS, StanceDataset, StanceDataEntry
from .._types import Stance

SemEval16Mapping: dict[str, Stance] = {
    "FAVOR": "Favor",
    "AGAINST": "Against",
    "NONE": "Neutral",
}


def _cast(data_entry) -> StanceDataEntry:
    r = cast(Mapping[str, Any], data_entry)
    stance = r["Stance"]
    _stance = SemEval16Mapping.get(stance, "Neutral")
    if stance is None:
        _stance = "Neutral"

    return {
        "stance": _stance,
        "target": r["Target"],
        "text": r["Tweet"],
    }


def load_sem16(split: SPLITS = "test") -> StanceDataset:
    """Load SEM16 dataset from Hugging Face and return a normalized dataset.
    We use the version https://huggingface.co/datasets/krishnagarg09/SemEval2016Task6
    """
    dataset = load_dataset("sem16", split=split)
    return StanceDataset([_cast(entry) for entry in dataset])
