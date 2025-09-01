import os
from openai import OpenAI
from returns.result import safe
from .prompts import INSTRUCTION, REWRITE, STEER
from ..datasets import StanceDataEntry, negate_stance
from ..utils import exp_retry


class Mutator:
    """
    The Mutator class is responsible for generating variations of text inputs
    using the OpenAI API.
    If the API request fails, it will return the original text.
    """

    def __init__(self, model: str = "gpt-4.1-nano"):

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable is not set"

        self.model = model
        self.client = OpenAI(api_key=api_key)

        self.mutators = [self.rewrite, self.steer]

    @safe
    @exp_retry
    def _gen(self, prompt: str) -> str:

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("No content generated")

        return content

    def rewrite(self, entry: StanceDataEntry) -> str:
        """rewrite mutator, the LLM rewrites the post without changing its meaning"""
        post = entry["text"]
        prompt = REWRITE.format(text=post)
        return self._gen(prompt).value_or(post)

    def steer(self, entry: StanceDataEntry) -> str:
        """steer mutator, if the request failed, return the original text"""
        post = entry["text"]
        stance = entry["stance"]
        prompt = STEER.format(
            text=post,
            stance=stance,
            target=entry["target"],
            direction=negate_stance(stance),
        )
        return self._gen(prompt).value_or(post)
