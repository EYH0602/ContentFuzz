import logging
from typing import Iterable
from enum import StrEnum

from google.genai.types import (
    GenerateContentResponse,
    Part,
)


def _get_text_from_parts(parts: Iterable[Part]) -> str | None:
    """Return concatenated text from parts, ignoring thoughts and non-text."""
    text = ""
    any_text_part_text = False

    for part in parts:
        if isinstance(part.text, str):
            if isinstance(part.thought, bool) and part.thought:
                continue
            any_text_part_text = True
            text += part.text

    return text if any_text_part_text else None


def get_texts(response: GenerateContentResponse) -> list[str]:
    """extract text from all candidates.
    This function is based on
    https://github.com/googleapis/python-genai/blob/b54208200eb3aefd13cd8199415682b146fce6df/google/genai/types.py#L6032
    """
    if not response.candidates:
        logging.warning("No candidate generated")
        return []

    texts: list[str] = []
    for idx, candidate in enumerate(response.candidates):
        if not candidate.content or not candidate.content.parts:
            logging.warning("Candidate %s has no content parts; skipping", idx)
            continue

        text = _get_text_from_parts(candidate.content.parts)

        if text is not None:
            texts.append(text)

    return texts


class FuzzerErr(StrEnum):
    """Error types during fuzzing"""

    NO_VALID_NEW_SEED = "No valid new seed is returned by the Mutator"
    FAILED_TO_MUTATE = "Failed to create an error triggering seed"
    EMPTY_SEED = "Seed population is empty"
