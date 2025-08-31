import os
from math import exp

from openai import OpenAI
from structured_logprobs import add_logprobs
from returns.result import safe

from ._base import AnalysisOutput, StanceOutput
from .utils import exp_retry


INSTRUCTION = """
You are a precise stance classifier.
Decide whether the author's attitude is Favor / Against / Neutral towards the target {target}.
Be conservative: if unclear, choose Neutral.
"""


class OpenAIAnalyzer:
    """Zero-shot stance analysis using OpenAI API"""

    def __init__(self, model: str = "gpt-4.1-nano"):

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable is not set"

        self.model = model
        self.client = OpenAI(api_key=api_key)

    @safe
    @exp_retry
    def analyze(self, text: str, target: str) -> AnalysisOutput:
        """Using OpenAI API to analyze the stance of a given text"""

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": INSTRUCTION.format(target=target),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            logprobs=True,
            response_format=StanceOutput,
            temperature=0,  # for reproduce
        )
        chat_completion = add_logprobs(completion)
        response = chat_completion.value
        probs = chat_completion.log_probs[0]
        # apply `exp` to all values of probs
        probs = {label: exp(logit) for label, logit in probs.items()}
        output = response.choices[0].message.content

        if output is None:
            raise ValueError("OpenAI API returned no output")

        stance = StanceOutput.model_validate_json(output)

        return stance, probs.get("label")
