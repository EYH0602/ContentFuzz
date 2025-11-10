from typing import Any, Mapping, cast

from datasets import load_dataset

from ._base import SPLITS, StanceDataset, StanceDataEntry
from .utils import remove_hash_tags
from .._types import Stance

SemEval16Mapping: dict[str, Stance] = {
    "FAVOR": "Favor",
    "AGAINST": "Against",
    "NONE": "Neutral",
}


def _cast(data_entry) -> StanceDataEntry:
    r = cast(Mapping[str, Any], data_entry)
    stance = r.get("Stance", "NONE")
    _stance = SemEval16Mapping.get(stance, "Neutral")

    text = r.get("Tweet", "").replace("#SemST", "")
    return {
        "stance": _stance,
        "target": r.get("Target", ""),
        "text": remove_hash_tags(text),
    }


def load_sem16(split: SPLITS = "test") -> StanceDataset:
    """Load SEM16 dataset from Hugging Face and return a normalized dataset.
    We use the version https://huggingface.co/datasets/krishnagarg09/SemEval2016Task6
    """
    dataset = load_dataset("krishnagarg09/SemEval2016Task6", split=split)
    return StanceDataset([_cast(entry) for entry in dataset])
