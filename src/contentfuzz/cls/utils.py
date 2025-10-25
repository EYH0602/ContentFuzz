from math import exp

from openai import OpenAI
from returns.result import safe

from ._base import AnalysisOutput, StanceOutput
from ..utils import exp_retry


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
    stance = completion.choices[0].message.content

    if stance is None or stance not in StanceOutput:
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
    "qwen3-next-80b-a3b-instruct": "qwen/qwen3-next-80b-a3b-instruct",
}
