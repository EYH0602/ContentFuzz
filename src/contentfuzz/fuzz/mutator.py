import os
import random

from openai import OpenAI

from .prompts import INSTRUCTION, REWRITE
from ..stance_dataset import StanceDataEntry
from ..utils import exp_retry


class Mutator:
    """
    The Mutator class is responsible for generating variations of text inputs
    using the OpenAI API.
    If the API request fails, it will return the original text.
    """

    def __init__(self, model: str = "gpt-4.1-nano", n: int = 5):

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable is not set"

        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.n = n

        # Temperature scheduling: 0.0 .. 2.0 (step 0.1) => 21 choices
        self.temperatures: list[float] = [round(0.1 * i, 1) for i in range(0, 21)]
        # Start with equal energies (uniform probability)
        self._energies: list[float] = [1.0 for _ in self.temperatures]
        self._last_temp_idx: int | None = None

    def _choose_temperature(self) -> float:
        # Weighted by energies; initially equal
        idx = random.choices(
            range(len(self.temperatures)), weights=self._energies, k=1
        )[0]
        self._last_temp_idx = idx
        return self.temperatures[idx]

    def update_energy(self, reward: float) -> None:
        """Increase energy of the last-used temperature by reward (e.g., m/n)."""
        if self._last_temp_idx is None:
            return
        # ensure non-negative reward
        reward = max(0.0, float(reward))
        self._energies[self._last_temp_idx] += reward

    @exp_retry
    def _gen(self, prompt: str) -> list[str]:
        temperature = self._choose_temperature()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            n=self.n,
            temperature=temperature,
        )

        contents = [
            choice.message.content
            for choice in completion.choices
            if choice.message.content
        ]

        return contents

    def rewrite(self, entry: StanceDataEntry) -> list[str]:
        """rewrite mutator, the LLM rewrites the post without changing its meaning"""
        post = entry["text"]
        prompt = REWRITE.format(text=post)
        return self._gen(prompt)

    def mutate(self, entry: StanceDataEntry) -> list[str]:
        """generates a list of mutated entries from the input entry"""
        texts = self.rewrite(entry)
        return texts
