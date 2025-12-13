import os
from math import exp
from typing import get_args

from deprecated import deprecated
from google import genai
from google.genai.client import AsyncClient
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    GenerateContentConfig,
    HttpOptions,
    ThinkingConfig,
)
from openai import OpenAI
from returns.future import future_safe
from returns.result import safe
from structured_logprobs import add_logprobs
from tenacity import (
    retry,
    wait_random_exponential,
)

from .._types import Stance, is_valid_stance
from ..utils import SEED, exp_retry
from ._base import AnalysisOutput, ClassifierOutput


@deprecated(reason="Use Google Gemini instead.")
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


@deprecated(reason="Use Google Gemini instead.")
def parse_reasoning_output(text: str, delim: str = "</think>") -> tuple[str, str]:
    """simple thinking content parser for Qwen models"""
    if delim not in text:
        return ("", text.strip())

    reasoning_part, answer_part = text.split(delim, 1)
    reasoning = reasoning_part.replace("<think>", "", 1).strip()
    answer = answer_part.strip()

    return reasoning, answer


@deprecated(reason="Use Google Gemini instead.")
@safe
@exp_retry
def classify_w_prob_openai(
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


def get_vertexai_client() -> genai.Client:
    """Create a google-genai client with Vertex AI API endpoints.
    Vertex AI is required to access logprobs from Gemini models.
    """
    client = genai.Client(
        vertexai=True,
        api_key=os.getenv("VERTEXAI_API_KEY"),
        http_options=HttpOptions(
            api_version="v1",
            headers={"X-Vertex-AI-LLM-Request-Type": "shared"},
        ),
    )
    return client


def _parse_gemini_prob(candidate) -> float | None:
    """Return probability computed from candidate logprobs if available."""
    logprobs_result = getattr(candidate, "logprobs_result", None)
    if not logprobs_result or not getattr(logprobs_result, "chosen_candidates", None):
        return None

    log_prob_sum = 0.0
    for chosen in logprobs_result.chosen_candidates:
        token_log_prob = getattr(chosen, "log_probability", None)
        if token_log_prob is None:
            return None
        log_prob_sum += token_log_prob
    return exp(log_prob_sum)


@safe
@exp_retry
def classify_w_prob(
    client: genai.Client,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
) -> AnalysisOutput:
    """Request a stance classification from Google Gemini with log probabilities."""

    response_schema = {
        "type": "STRING",
        "enum": get_args(Stance),
    }
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=GenerateContentConfig(
            temperature=0,
            system_instruction=system_prompt,
            response_mime_type="text/x.enum",
            response_schema=response_schema,
            response_logprobs=True,
            logprobs=1,
            seed=SEED,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            thinking_config=ThinkingConfig(
                thinking_budget=0,  # disable thinking
            ),
        ),
    )

    if not response.candidates:
        raise ValueError("Gemini API returned no candidates")

    candidate = response.candidates[0]
    stance = response.text

    if not stance or not is_valid_stance(stance):
        raise ValueError(f"Invalid stance output: {stance}")

    return stance, _parse_gemini_prob(candidate)


@future_safe
@retry(
    reraise=True,
    wait=wait_random_exponential(max=60, multiplier=1),
)
async def classify_w_prob_async(
    client: AsyncClient,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
) -> AnalysisOutput:
    """Request a stance classification from Google Gemini with log probabilities.
    This function is the async version of `classify_w_prob`.
    """

    response_schema = {
        "type": "STRING",
        "enum": get_args(Stance),
    }
    response = await client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=GenerateContentConfig(
            temperature=0,
            system_instruction=system_prompt,
            response_mime_type="text/x.enum",
            response_schema=response_schema,
            response_logprobs=True,
            logprobs=1,
            seed=SEED,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            thinking_config=ThinkingConfig(
                thinking_budget=0,  # disable thinking
            ),
        ),
    )

    if not response.candidates:
        raise ValueError("Gemini API returned no candidates")

    candidate = response.candidates[0]
    stance = response.text

    if not stance or not is_valid_stance(stance):
        raise ValueError(f"Invalid stance output: {stance}")

    return stance, _parse_gemini_prob(candidate)
