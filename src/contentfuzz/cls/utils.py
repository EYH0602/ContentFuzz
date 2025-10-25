from math import exp
import os

from openai import OpenAI
from returns.result import safe

from ._base import AnalysisOutput, StanceOutput
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
    return model_name, client


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

    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        logprobs=True,
        temperature=0,  # for reproduce
    )
    content = completion.choices[0].message.content

    if content is None:
        raise ValueError("API no output")

    rationale, stance = parse_reasoning_output(content)
    if stance not in StanceOutput:
        raise ValueError(f"StanceOutput is invalid: {stance}")

    # stance = StanceOutput.model_validate_json(output)
    logprobs = completion.choices[0].logprobs
    if not logprobs or not logprobs.content:
        raise ValueError("logprobs is None")
    # token_logprobs = [tp.logprob for tp in resp.output[0].content[0].token_logprobs]
    token_logprobs = [c.logprob for c in logprobs.content]
    prob: float = sum(token_logprobs)  # the total likely-hood

    return stance, exp(prob)


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
