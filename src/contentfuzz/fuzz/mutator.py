import os
import random

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
    GenerateContentResponse,
    AutomaticFunctionCallingConfig,
)

from .prompts import INSTRUCTION, REWRITE
from .utils import get_texts
from ..stance_dataset import StanceDataEntry
from ..utils import exp_retry, SEED


class Mutator:
    """
    Generates variations of input texts using the OpenAI Chat Completions API.

    Parameters
    - model: str
      The OpenAI chat model to use. Defaults to "gpt-4.1-nano".

    - n: int
      Number of completions to request per prompt. Defaults to 5.

    - temperature: float | None
      Controls randomness of generation. If None (default), enable temperature
      scheduling where a temperature is sampled from a discrete range [0.0, 2.0]
      and adapted via a simple reward-based energy update. If set to a float,
      scheduling is disabled and the fixed value is used for all generations.

    Notes
    - Requires the `OPENAI_API_KEY` environment variable to be set.
    - When scheduling is enabled, `update_energy` adjusts sampling weights based
      on observed success to bias future temperature choices.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        n: int = 5,
        temperature: float | None = None,
    ):
        assert 1 <= n <= 8, "n in [1,8] is supported"

        api_key = os.getenv("GEMINI_API_KEY")
        assert api_key is not None, "GEMINI_API_KEY environment variable is not set"

        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.n = n
        # If temperature is None, enable scheduling; otherwise use fixed temperature
        self.use_temp_schedule: bool = temperature is None
        self.fixed_temperature: float | None = (
            None if temperature is None else float(temperature)
        )

        # Temperature scheduling state
        if self.use_temp_schedule:
            # 0.0 .. 2.0 (step 0.1) => 21 choices
            self.temperatures: list[float] = [round(0.1 * i, 1) for i in range(0, 21)]
            # Start with equal energies (uniform probability)
            self._energies: list[float] = [1.0 for _ in self.temperatures]
            self._last_temp_idx: int | None = None
        else:
            # Placeholders to keep attributes available if referenced
            # Use the provided fixed temperature
            assert self.fixed_temperature is not None
            self.temperatures = [self.fixed_temperature]
            self._energies = [1.0]
            self._last_temp_idx = None

    def _choose_temperature(self) -> float:
        # If scheduling disabled, always use the fixed base temperature
        if not self.use_temp_schedule:
            assert self.fixed_temperature is not None
            return self.fixed_temperature
        # Weighted by energies; initially equal
        idx = random.choices(
            range(len(self.temperatures)), weights=self._energies, k=1
        )[0]
        self._last_temp_idx = idx
        return self.temperatures[idx]

    def update_energy(self, reward: float) -> None:
        """Increase energy of the last-used temperature by reward (e.g., m/n).

        No-op when temperature scheduling is disabled.
        """
        if not self.use_temp_schedule:
            return
        if self._last_temp_idx is None:
            return
        # ensure non-negative reward
        reward = max(0.0, float(reward))
        self._energies[self._last_temp_idx] += reward

    @exp_retry
    def _gen(self, prompt: str) -> list[str]:
        temperature = self._choose_temperature()
        response: GenerateContentResponse = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCTION,
                thinking_config=ThinkingConfig(
                    thinking_budget=0,
                ),
                temperature=temperature,
                candidate_count=self.n,
                seed=SEED,
                automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        return get_texts(response)

    def rewrite(self, entry: StanceDataEntry) -> list[str]:
        """rewrite mutator, the LLM rewrites the post without changing its meaning"""
        post = entry["text"]
        prompt = REWRITE.format(text=post)
        return self._gen(prompt)

    def mutate(self, entry: StanceDataEntry) -> list[str]:
        """generates a list of mutated entries from the input entry"""
        texts = self.rewrite(entry)
        return texts
