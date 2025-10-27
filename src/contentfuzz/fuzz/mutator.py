import os
import random

from openai import OpenAI

from .prompts import INSTRUCTION, REWRITE, STEER, TLDR, TAGS
from ..stance_dataset import StanceDataEntry, negate_stance
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

    @exp_retry
    def _gen(self, prompt: str) -> list[str]:

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            n=self.n,
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
