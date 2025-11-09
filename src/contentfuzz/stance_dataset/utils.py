from .._types import Stance


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
