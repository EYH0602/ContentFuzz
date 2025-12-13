import asyncio

from google.genai.client import AsyncClient
from returns.result import ResultE, Success
from tqdm import tqdm

from ._base import AnalysisOutput
from .utils import classify_w_prob, classify_w_prob_async, get_vertexai_client

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

    def analyze(self, text: str, target: str) -> ResultE[AnalysisOutput]:
        """Using OpenAI API to analyze the stance of a given text"""
        return classify_w_prob(
            self.client,
            self.model,
            INSTRUCTION.format(target=target),
            text,
        )

    async def _run_async_analysis(
        self,
        tasks: list[tuple[str, str]],
        async_client: AsyncClient,
        batch_size: int,
    ) -> list[AnalysisOutput]:
        sem = asyncio.Semaphore(batch_size)

        async def handle_task(task: tuple[str, str]) -> AnalysisOutput:
            text, target = task
            async with sem:
                result = await classify_w_prob_async(
                    async_client,
                    self.model,
                    INSTRUCTION.format(target=target),
                    text,
                )
                return result

        results = await asyncio.gather(*(handle_task(t) for t in tasks))
        return results

    def batched_analysis(
        self, tasks: list[tuple[str, str]], batch_size: int = 1
    ) -> list[ResultE[AnalysisOutput]]:
        """Run async zero-shot analysis in batches using a single event loop.
        Args:
            tasks (list[tuple[str, str]]): List of (text, target) pairs
            batch_size (int, optional): Number of samples to process concurrently. Defaults to 1.
        Returns:
            list[ResultE[AnalysisOutput]]: List of analysis results in order
        """

        async def _run_batches() -> list[AnalysisOutput]:
            # Use one async client for the whole run to avoid re-inits.
            async_client = self.client.aio
            try:
                return await self._run_async_analysis(tasks, async_client, batch_size)
            finally:
                await async_client.aclose()

        batch_outputs = asyncio.run(_run_batches())
        return [Success(output) for output in batch_outputs]
