import os
import random

from openai import OpenAI

from .prompts import INSTRUCTION, REWRITE, STEER, TLDR, TAGS
from ..datasets import StanceDataEntry, negate_stance
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

        self.mutators = [
            self.rewrite,
            self.steer,
            self.tldr,
            self.tags,
        ]

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

    def steer(self, entry: StanceDataEntry) -> list[str]:
        """steer mutator, if the request failed, return the original text"""
        post = entry["text"]
        stance = entry["stance"]
        prompt = STEER.format(
            text=post,
            stance=stance,
            target=entry["target"],
            direction=negate_stance(stance),
        )
        return self._gen(prompt)

    def tldr(self, entry: StanceDataEntry) -> list[str]:
        """TL;DR mutator
        the LLM generates a summary of the post with slightly opposite stance,
        which is added to the front of the original post
        """
        post = entry["text"]
        stance = entry["stance"]
        prompt = TLDR.format(
            text=post,
            stance=stance,
            target=entry["target"],
            direction=negate_stance(stance),
        )
        return [f"{tldr}\n\n{post}" for tldr in self._gen(prompt)]

    def tags(self, entry: StanceDataEntry) -> list[str]:
        """Hash-Tags mutator
        the LLM generates five hash-tags for the post,
        among which two are slightly opposite to the original stance.
        The hash-tags are added to the end of the post.
        """
        post = entry["text"]
        stance = entry["stance"]
        prompt = TAGS.format(
            text=post,
            stance=stance,
            target=entry["target"],
            direction=negate_stance(stance),
        )

        return [f"{post}\n\n{tags}" for tags in self._gen(prompt)]

    def mutate(self, entry: StanceDataEntry) -> list[str]:
        """generated a list of mutated entries from the input entry"""
        mutator = random.choice(self.mutators)
        texts = mutator(entry)
        return texts
