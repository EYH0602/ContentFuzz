from functools import wraps
from typing import Callable, ParamSpec, TypeVar
import os

from tenacity import retry, stop_after_attempt, wait_random_exponential

P = ParamSpec("P")
R = TypeVar("R")


def exp_retry(func: Callable[P, R]) -> Callable[P, R]:
    """Apply exponential backoff (min=1, max=10) for 3 attempts.
    Type-preserving and exception-preserving.
    """

    @wraps(func)
    @retry(
        reraise=True,
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapper


SEED = int(os.getenv("SEED", "0"))


def get_default_cls_output_path(
    output_dir: str, dataset_name: str, analyzer_name: str, model: str
) -> str:
    """construct default classification output path"""
    return (
        f"{output_dir}/{analyzer_name}+{model.replace("/", "--")}+{dataset_name}.jsonl"
    )


def get_default_atk_output_path(
    output_dir: str, cls_output_path: str, attack_model: str
) -> str:
    """construct a default attack output path based on cls_output_path"""
    cls_output_basename = os.path.basename(cls_output_path)
    cls_output_filename = os.path.splitext(cls_output_basename)[0]
    return f"{output_dir}/{cls_output_filename}={attack_model}.jsonl"
