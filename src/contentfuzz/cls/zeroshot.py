import os
from typing import Literal
from math import exp
from pydantic import BaseModel

from openai import OpenAI
from structured_logprobs import add_logprobs

from ._base import AnalysisOutput

Label = Literal["Favor", "Against", "Neutral"]

INSTRUCTION = """
You are a precise stance classifier.
Decide whether the author's attitude is Favor / Against / Neutral.
Be conservative: if unclear, choose Neutral.
Return the JSON object only.
"""


class Stance(BaseModel):
    """Classifier response format"""

    label: Label
    rationale: str


class OpenAIAnalyzer:
    """Zero-shot stance analysis using OpenAI API"""

    def __init__(self, model: str = "gpt-4.1-nano"):

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable is not set"

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def analyze(self, text: str) -> AnalysisOutput:
        """Using OpenAI API to analyze the stance of a given text"""

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": INSTRUCTION.strip(),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            logprobs=True,
            response_format=Stance,
        )
        chat_completion = add_logprobs(completion)
        response = chat_completion.value
        probs = chat_completion.log_probs[0]
        # apply `exp` to all values of probs
        probs = {label: exp(logit) for label, logit in probs.items()}
        output = response.choices[0].message.content

        if output is None:
            raise ValueError("OpenAI API returned no output")
        stance = Stance.model_validate_json(output)

        return AnalysisOutput(
            stance=stance.label,
            rationale=stance.rationale,
            prob=probs.get("label", None),
        )
