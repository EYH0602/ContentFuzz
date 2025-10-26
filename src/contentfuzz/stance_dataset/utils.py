from ._base import StanceDataEntry
from .._types import Stance

CHN_TO_EN: dict[str, Stance] = {"支持": "Favor", "反对": "Against", "中立": "Neutral"}


def translate_stance(stance: str) -> Stance:
    """unify stance language
    1. Chinese -> English
    2. Irrelevant -> Neutral
    """

    _stance: Stance = CHN_TO_EN.get(stance, "Neutral")
    if stance == "Irrelevant":
        _stance = "Neutral"
    return _stance


def load_data_entry(data: dict) -> StanceDataEntry:
    """Load a data entry from a dictionary."""
    return StanceDataEntry(
        text=data["text"],
        target=data["target"],
        stance=translate_stance(data["stance"]),
    )


def negate_stance(stance: Stance, neutral_to: Stance | None = None) -> Stance:
    """Negate the stance.
    For example:
        - Favor -> Against
        - Against -> Favor
        - Neutral -> `neutral_to` if it is specified, or "Favor"
    """
    if stance == "Favor":
        return "Against"
    elif stance == "Against":
        return "Favor"

    return neutral_to or "Favor"
