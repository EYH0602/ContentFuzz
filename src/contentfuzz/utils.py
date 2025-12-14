import os
from typing import Any, Callable, Literal, ParamSpec, TypeVar, cast

from google.genai._interactions import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_never,
    wait_random_exponential,
)

from .env import MAX_RETRIES

# ISO 639-1 language code
# https://en.wikipedia.org/wiki/ISO_639-1
Language = Literal["en", "zh"]


P = ParamSpec("P")
R = TypeVar("R")

RetryExceptions = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    TimeoutError,
)


retry_kwargs: dict[str, Any] = dict(
    reraise=True,
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(RetryExceptions),
    stop=stop_after_attempt(MAX_RETRIES) if MAX_RETRIES else stop_never,
)


SEED = int(os.getenv("SEED", "0"))


def exp_retry(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for retrying API calls to all LLMs with exponential backoff.

    If the environment variable `MAX_RETRIES` is not set, retries indefinitely.
    This wrapper preserves type information that would otherwise be lost when using tenacity's retry decorator.
    """
    decorated = retry(**retry_kwargs)(func)  # mypy thinks this returns Any
    return cast(Callable[P, R], decorated)


def get_default_cls_output_path(
    output_dir: str, dataset_name: str, analyzer_name: str, model: str
) -> str:
    """construct default classification output path"""
    return (
        f"{output_dir}/{analyzer_name}+{model.replace('/', '--')}+{dataset_name}.jsonl"
    )


def get_default_atk_output_path(
    output_dir: str, cls_output_path: str, attack_model: str
) -> str:
    """construct a default attack output path based on cls_output_path"""
    cls_output_basename = os.path.basename(cls_output_path)
    cls_output_filename = os.path.splitext(cls_output_basename)[0]
    return f"{output_dir}/{cls_output_filename}={attack_model}.jsonl"


def get_skip_cnt(file_path: str) -> int:
    """count number of a record JSONL file"""
    if not os.path.isfile(file_path):
        return 0
    with open(file_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines
