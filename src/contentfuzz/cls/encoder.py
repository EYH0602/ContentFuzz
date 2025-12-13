from transformers import AutoModelForSequenceClassification, AutoTokenizer
from returns.result import safe
import torch

from .._types import Stance
from ._base import AnalysisOutput


def to_prompt(text: str, target: str) -> str:
    """combine post text and target to prompt as as fine-tuning"""
    return f"Text: {text} \nTarget: {target}\n"


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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForSequenceClassification.from_pretrained(model).to(
            self._device
        )
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
        # Tokenize as a sentence pair for stance classification
        # tokenization setting is the same as fine-tuning
        batch = self._tokenizer(
            to_prompt(text, target),
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(self._device) for key, value in batch.items()}

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

    @safe
    def analyze_multiple(
        self, entries: list[tuple[str, str]], batch_size: int = 8
    ) -> list[AnalysisOutput]:
        """Batch analysis for multiple `(text, target)` pairs."""
        results: list[AnalysisOutput] = []
        id2label: dict[int, str] = getattr(self._model.config, "id2label", {}) or {}

        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            prompts = [to_prompt(text, target) for text, target in chunk]
            batch = self._tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            batch = {key: value.to(self._device) for key, value in batch.items()}

            with torch.no_grad():
                logits = self._model(**batch).logits
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.max(probs, dim=-1)

            for idx_tensor, prob_tensor in zip(top_indices, top_probs):
                idx = int(idx_tensor.item())
                prob = float(prob_tensor.item())

                stance: Stance | None = self._id2stance.get(idx)
                if stance is None:
                    label = id2label.get(idx, "")
                    stance = _label_to_stance(label) or "Neutral"

                results.append((stance, prob))

        return results


def _label_to_stance(label: str) -> Stance | None:
    """Map model label names to Stance values by simple heuristics.

    Accepts common variants like "favor", "support", "against", "oppose",
    and "neutral"/"irrelevant" (case-insensitive).
    Returns None if it cannot confidently determine a mapping.
    """
    label = label.lower()
    if any(k in label for k in ("favor", "support", "pro", "label_0", "支持")):
        return "Favor"
    if any(k in label for k in ("against", "oppose", "anti", "con", "label_1", "反对")):
        return "Against"
    if any(k in label for k in ("neutral", "irrelevant", "label_2", "中立")):
        return "Neutral"
    # Hugging Face default labels (LABEL_0, LABEL_1, LABEL_2) are ambiguous
    return None
