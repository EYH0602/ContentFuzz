from typing import TypedDict, Literal
from .._types import Stance


class StanceDataEntry(TypedDict, total=True):
    """basic structure of data entry in any stance analysis dataset"""

    text: str
    target: str
    stance: Stance


StanceDataset = list[StanceDataEntry]

SPLITS = Literal["train", "test", "validation"]
