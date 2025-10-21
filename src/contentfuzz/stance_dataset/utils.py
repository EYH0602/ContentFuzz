from ._base import StanceDataEntry

CHN_TO_EN = {"支持": "Favor", "反对": "Against", "中立": "Neutral"}


def translate_stance(stance: str) -> str:
    """unify stance language
    1. Chinese -> English
    2. Irrelevant -> Neutral
    """

    stance = CHN_TO_EN.get(stance, stance)
    if stance == "Irrelevant":
        stance = "Neutral"
    return stance


def load_data_entry(data: dict) -> StanceDataEntry:
    """Load a data entry from a dictionary."""
    return StanceDataEntry(
        text=data["text"],
        target=data["target"],
        stance=translate_stance(data["stance"]),
    )


def negate_stance(stance: str, neutral_to: str | None = None) -> str:
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
