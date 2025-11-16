from typing import Any, Mapping, cast

from datasets import load_dataset

from ._base import SPLITS, StanceDataset, StanceDataEntry
from .utils import remove_hash_tags
from .._types import Stance

# https://github.com/emilyallaway/zero-shot-stance/blob/e7c4775182b184730f350995f8260579c9e066fe/data/README.md
# label: stance label, 0=con, 1=pro, 2=neutral
VASTMapping: dict[int, Stance] = {
    0: "Against",
    1: "Favor",
    2: "Neutral",
}


def _cast(data_entry) -> StanceDataEntry:
    r = cast(Mapping[str, Any], data_entry)
    label = r.get("label", 2)
    stance = VASTMapping.get(label, "Neutral")

    text = r.get("post", "")
    return {
        "stance": stance,
        "target": r.get("topic_str", ""),
        "text": remove_hash_tags(text),
    }


def load_sem16(split: SPLITS = "test") -> StanceDataset:
    """Load SEM16 dataset from Hugging Face and return a normalized dataset.
    We use the version https://huggingface.co/datasets/krishnagarg09/SemEval2016Task6
    """
    dataset = load_dataset("yfhe/VAST", split=split)
    return StanceDataset([_cast(entry) for entry in dataset])
