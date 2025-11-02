from transformers import AutoModelForSequenceClassification, AutoTokenizer
from returns.result import safe
import torch

from .._types import Stance
from ._base import AnalysisOutput


class Encoder:
    """BERT-style encoder classifier (forward-only).

    - Tokenizes `(text, target)` as a sentence pair.
    - Runs a single forward pass through a sequence classification head.
    - Applies softmax to logits and selects the highest-prob class.
    - Returns `(stance, probability_of_that_choice)`.
    """

    def __init__(self, model: str):
        # Protocol requires `model: str` field
        self.model = model
        # Lazy-load tokenizer/model so construction is lightweight in notebooks/CLIs
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._model.eval()

        # Build id->stance mapping from model labels when possible
        self._id2stance: dict[int, Stance] = {}
        id2label: dict[int, str] = getattr(self._model.config, "id2label", {}) or {}
        if id2label:
            for idx, label in id2label.items():
                stance = _label_to_stance(label)
                # Only register if we can confidently map the label
                if stance is not None:
                    self._id2stance[idx] = stance

    @safe
    def analyze(self, text: str, target: str) -> AnalysisOutput:
        """Run a forward pass and return stance with probability.

        - Tokenizes `(text, target)` as a sentence pair for the finetuned
          sequence classification head.
        - Applies softmax to logits and selects the top label, mapped to
          `Stance` when possible.
        - Returns a tuple `(stance, probability)` where probability is the
          model's softmax score for the chosen stance.

        Note: this method is decorated with `@safe`, so callers receive a
        `Result` wrapping the output or an exception.
        """
        # todo: process text + target the same format as fine-tuning
        # Tokenize as a sentence pair for stance classification
        batch = self._tokenizer(
            text,
            target,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**batch)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(probs).item())
            prob = float(probs[idx].item())

        # Resolve stance label
        stance: Stance | None = self._id2stance.get(idx)
        if stance is None:
            # Fall back to interpreting label string if available
            id2label: dict[int, str] = getattr(self._model.config, "id2label", {}) or {}
            label = id2label.get(idx, "")
            stance = _label_to_stance(label) or "Neutral"

        return stance, prob


def _label_to_stance(label: str) -> Stance | None:
    """Map model label names to Stance values by simple heuristics.

    Accepts common variants like "favor", "support", "against", "oppose",
    and "neutral"/"irrelevant" (case-insensitive).
    Returns None if it cannot confidently determine a mapping.
    """
    l = label.lower()
    if any(k in l for k in ("favor", "support", "pro")):
        return "Favor"
    if any(k in l for k in ("against", "oppose", "anti", "con")):
        return "Against"
    if any(k in l for k in ("neutral", "irrelevant")):
        return "Neutral"
    # Hugging Face default labels (LABEL_0, LABEL_1, LABEL_2) are ambiguous
    return None
