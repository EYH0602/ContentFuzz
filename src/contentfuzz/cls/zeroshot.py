from .utils import classify_w_prob, get_vertexai_client


INSTRUCTION = """
You are a precise stance classifier.
Decide whether the author's attitude is Favor / Against / Neutral towards the target {target}.
Be conservative: if unclear, choose Neutral.
ONLY output one word chosen from Favor, Against, Neutral.
"""


class ZeroshotAnalyzer:
    """Zero-shot stance analysis using OpenAI API"""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        # self.model, self.client = _get_model_and_client(model)
        self.model = model
        self.client = get_vertexai_client()

    def analyze(self, text: str, target: str):
        """Using OpenAI API to analyze the stance of a given text"""
        return classify_w_prob(
            self.client,
            self.model,
            INSTRUCTION.format(target=target),
            text,
        )
