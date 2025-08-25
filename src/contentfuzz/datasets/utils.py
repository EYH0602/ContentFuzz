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
