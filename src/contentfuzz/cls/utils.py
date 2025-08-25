from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


def exp_retry(func):
    """wrapper to apply exponential backoff retry
    with min = 1 and max = 10 for 3 attempts
    """

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
