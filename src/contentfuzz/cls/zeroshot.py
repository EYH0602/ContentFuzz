import os

from openai import OpenAI

from .utils import classify_w_prob, MODEL_NAME_MAP


INSTRUCTION = """
You are a precise stance classifier.
Decide whether the author's attitude is Favor / Against / Neutral towards the target {target}.
Be conservative: if unclear, choose Neutral.
ONLY output one word chosen from Favor, Against, Neutral.
"""


class OpenAIAnalyzer:
    """Zero-shot stance analysis using OpenAI API"""

    def __init__(self, model: str = "gpt-4.1-nano"):

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable is not set"

        use_ppio = model in MODEL_NAME_MAP
        self.model = MODEL_NAME_MAP[model] if use_ppio else model
        self.client = OpenAI(
            base_url="https://api.ppinfra.com/openai" if use_ppio else None,
            api_key=api_key,
        )

    def analyze(self, text: str, target: str):
        """Using OpenAI API to analyze the stance of a given text"""
        return classify_w_prob(
            self.client,
            self.model,
            INSTRUCTION.format(target=target),
            text,
        )
