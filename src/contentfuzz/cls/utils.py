from math import exp
import os

from openai import OpenAI
from returns.result import safe
from structured_logprobs import add_logprobs

from ._base import AnalysisOutput, ClassifierOutput
from ..utils import exp_retry


def _get_model_and_client(model_name: str) -> tuple[str, OpenAI]:
    use_ppio = model_name in MODEL_NAME_MAP
    key_name = "PPIO_API_KEY" if use_ppio else "OPENAI_API_KEY"
    api_key = os.getenv(key_name)
    assert api_key is not None, "The API key environment variable is not set"

    model = MODEL_NAME_MAP[model_name] if use_ppio else model_name
    client = OpenAI(
        base_url="https://api.ppinfra.com/openai" if use_ppio else None,
        api_key=api_key,
    )
    return model, client


def parse_reasoning_output(text: str, delim: str = "</think>") -> tuple[str, str]:
    """simple thinking content parser for Qwen models"""
    if delim not in text:
        return ("", text.strip())

    reasoning_part, answer_part = text.split(delim, 1)
    reasoning = reasoning_part.replace("<think>", "", 1).strip()
    answer = answer_part.strip()

    return reasoning, answer


@safe
@exp_retry
def classify_w_prob(
    client: OpenAI,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
) -> AnalysisOutput:
    """request to OpenAI client with logprob"""
    messages = (
        [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        if system_prompt
        else []
    )
    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        },
    )

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,  # type: ignore
        logprobs=True,
        response_format=ClassifierOutput,
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

    output = ClassifierOutput.model_validate_json(output)

    return output.stance, probs.get("stance")


MODEL_NAME_MAP: dict[str, str] = {
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3.2-exp": "deepseek/deepseek-v3.2-exp",
    "qwen3-next-80b-a3b-instruct": "qwen/qwen3-next-80b-a3b-instruct",
    "qwen3-235b-a22b-fp8": "qwen/qwen3-235b-a22b-fp8",
    "qwen3-235b-a22b-instruct-2507": "qwen/qwen3-235b-a22b-instruct-2507",
    "qwen3-235b-a22b-thinking-2507": "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen3-4b-fp8": "qwen/qwen3-4b-fp8",
    "qwen3-30b-a3b-fp8": "qwen/qwen3-30b-a3b-fp8",
    "deepseek-r1-0528-qwen3-8b": "deepseek/deepseek-r1-0528-qwen3-8b",
}
