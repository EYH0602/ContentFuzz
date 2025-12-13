import asyncio

from google.genai.client import AsyncClient
from returns.result import ResultE

from ._base import AnalysisOutput
from .utils import classify_w_prob_async, get_vertexai_client

INSTRUCTION = """
You are a precise stance classifier.
Decide whether the author's attitude is Favor / Against / Neutral towards the target {target}.
Be conservative: if unclear, choose Neutral.
ONLY output one word chosen from Favor, Against, Neutral.
"""


class ZeroshotAnalyzer:
    """Zero-shot stance analysis using OpenAI API"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
    ):
        # self.model, self.client = _get_model_and_client(model)
        self.model = model
        self.client = get_vertexai_client()

    def analyze(
        self, tasks: list[tuple[str, str]], batch_size: int | None = None
    ) -> list[ResultE[AnalysisOutput]]:
        """Run async zero-shot analysis in batches using a single event loop.
        Args:
            tasks (list[tuple[str, str]]): List of (text, target) pairs
            batch_size (int, optional): Number of samples to process concurrently.
                Defaults to None.
                If None, processes all samples concurrently, and let retry handle rate limits.
        Returns:
            list[ResultE[AnalysisOutput]]: List of analysis results in order
        """

        async def _run_batches() -> list[ResultE[AnalysisOutput]]:
            # Use one async client for the whole run to avoid re-inits.
            async_client = self.client.aio
            sem = asyncio.Semaphore(batch_size or len(tasks))

            # Use one async client for the whole run to avoid re-inits.
            async def run_task(task):
                text, target = task
                async with sem:
                    result = await classify_w_prob_async(
                        async_client,
                        self.model,
                        INSTRUCTION.format(target=target),
                        text,
                    )
                    return result._inner_value

            try:
                return await asyncio.gather(*(run_task(task) for task in tasks))
            finally:
                await async_client.aclose()

        return asyncio.run(_run_batches())
