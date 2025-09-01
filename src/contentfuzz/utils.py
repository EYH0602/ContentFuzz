from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from tenacity import retry, stop_after_attempt, wait_random_exponential

P = ParamSpec("P")
R = TypeVar("R")


def exp_retry(func: Callable[P, R]) -> Callable[P, R]:
    """Apply exponential backoff (min=1, max=10) for 3 attempts, type-preserving."""

    @wraps(func)
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapper
