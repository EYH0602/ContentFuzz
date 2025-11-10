import re

from .._types import Stance

_HASHTAG_PATTERN = re.compile(r"#(?:\w+#)+|#\w+")
_SENTENCE_END_CHARS = set(".!?;:…")
_TRAILING_ENCLOSERS = "\"'’”)]}>"


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


def remove_hash_tags(text: str) -> str:
    """Remove hash tags from the text."""
    def _has_context(segment: str) -> bool:
        # Drop hashtags in the segment to see if any real text remains.
        return bool(_HASHTAG_PATTERN.sub("", segment).strip())

    def _replace(match: re.Match[str]) -> str:
        token = match.group()
        normalized = token.replace("#", "").lower()
        if normalized == "semst":
            # Always drop the SemST marker regardless of position.
            return ""

        start, end = match.span()
        # Determine whether there is real text on both sides (i.e. hashtag is inside).
        has_text_before = _has_context(text[:start])
        has_text_after = _has_context(text[end:])

        if has_text_before and has_text_after:
            # Keep inner hashtags but drop the hash markers, preserving multi-word tags.
            parts = [chunk for chunk in match.group().split("#") if chunk]
            return " ".join(parts)

        if not has_text_before:
            # Drop hashtags at the very beginning.
            return ""

        if not has_text_after:
            # Consider a hashtag trailing only if the sentence right before ends with punctuation.
            before_trim = _HASHTAG_PATTERN.sub("", text[:start]).rstrip()
            # Ignore closing quotes/brackets when checking for punctuation.
            while before_trim and before_trim[-1] in _TRAILING_ENCLOSERS:
                before_trim = before_trim[:-1].rstrip()
            if before_trim and before_trim[-1] in _SENTENCE_END_CHARS:
                return ""

        # Keep hashtags that function as inline words; strip only the markers.
        parts = [chunk for chunk in token.split("#") if chunk]
        return " ".join(parts)

    # Replace hashtags contextually, either stripping markers or removing them.
    cleaned = _HASHTAG_PATTERN.sub(_replace, text)
    # Collapse any double spaces left behind so words stay properly separated.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Trim whitespace that can appear when a hashtag begins or ends the sentence.
    cleaned = re.sub(r"[ \t]+$", "", cleaned)
    cleaned = re.sub(r"^[ \t]+", "", cleaned)
    return cleaned
