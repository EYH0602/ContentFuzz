from dataclasses import dataclass


@dataclass
class StanceDataEntry:
    """basic structure of data entry in any stance analysis dataset"""

    text: str
    target: str
    stance: str

    domain: str | None = None  # optional domain
