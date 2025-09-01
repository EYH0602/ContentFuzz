from typing import TypedDict


class StanceDataEntry(TypedDict, total=True):
    """basic structure of data entry in any stance analysis dataset"""

    text: str
    target: str
    stance: str
