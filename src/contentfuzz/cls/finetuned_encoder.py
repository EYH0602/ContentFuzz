from transformers import pipeline
from .utils import classify_w_prob, _get_model_and_client


class FinetunedEncoder:
    def __init__(self, model_name: str = "hfl/chinese-macbert-base"):
        self.model = model_name
