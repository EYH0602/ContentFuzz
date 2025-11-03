from typing import Literal, TypeGuard, get_args

Stance = Literal["Favor", "Against", "Neutral"]


def is_valid_stance(stance: str) -> TypeGuard[Stance]:
    """check if a string stance is of `Stance` type"""
    return stance in get_args(Stance)
