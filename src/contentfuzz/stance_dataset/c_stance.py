from typing import Any, Mapping, cast, Literal

from datasets import load_dataset

from .utils import translate_stance
from ._base import SPLITS, StanceDataset, StanceDataEntry


C_STANCE = Literal["c-stance-a", "c-stance-b"]
DATASET_NAME = C_STANCE


def _hf_repo_id(name: C_STANCE) -> str:
    return f"yfhe/{name.upper()}"


def _cast(data_entry) -> StanceDataEntry:
    r = cast(Mapping[str, Any], data_entry)
    return {
        "stance": translate_stance(r["Stance"]),
        "target": r["Target"],
        "text": r["Text"],
    }


def load_c_stance(name: C_STANCE, split: SPLITS = "test") -> StanceDataset:
    """Load C-STANCE dataset from Hugging Face and return a normalized dataset.

    The returned DataFrame has columns: ["text", "target", "stance"].
    Stance values are normalized to Favor/Against/Neutral.
    """
    ds = load_dataset(_hf_repo_id(name), split=split)
    return [_cast(d) for d in ds]
