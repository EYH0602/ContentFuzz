from typing import Any, Mapping, cast, Literal

from datasets import load_dataset


from .._types import Stance
from ._base import SPLITS, StanceDataset, StanceDataEntry


CSTANCEChoices = Literal["c-stance-a", "c-stance-b"]

CHN_TO_EN: dict[str, Stance] = {"支持": "Favor", "反对": "Against", "中立": "Neutral"}


def _hf_repo_id(name: CSTANCEChoices) -> str:
    return f"yfhe/{name.upper()}"


def _cast(data_entry) -> StanceDataEntry:
    r = cast(Mapping[str, Any], data_entry)
    stance = r["Stance"]
    _stance: Stance = CHN_TO_EN.get(stance, "Neutral")
    if stance == "Irrelevant":
        _stance = "Neutral"

    return {
        "stance": _stance,
        "target": r["Target"],
        "text": r["Text"],
    }


def load_c_stance(name: CSTANCEChoices, split: SPLITS = "test") -> StanceDataset:
    """Load C-STANCE dataset from Hugging Face and return a normalized dataset.

    The returned DataFrame has columns: ["text", "target", "stance"].
    Stance values are normalized to Favor/Against/Neutral.
    """
    ds = load_dataset(_hf_repo_id(name), split=split)
    return [_cast(d) for d in ds]
