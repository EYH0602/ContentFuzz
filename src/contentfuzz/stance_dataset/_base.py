from typing import TypedDict, Literal


class StanceDataEntry(TypedDict, total=True):
    """basic structure of data entry in any stance analysis dataset"""

    text: str
    target: str
    stance: str


StanceDataset = list[StanceDataEntry]

SPLITS = Literal["train", "test", "validation"]
