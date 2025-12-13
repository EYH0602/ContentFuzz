import asyncio
from multiprocessing import cpu_count

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

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
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
        self, tasks: list[tuple[str, str]], async_client: AsyncClient
    ):
        sem = asyncio.Semaphore(cpu_count())

        async def handle_task(task: tuple[str, str]) -> AnalysisOutput:
            text, target = task
            async with sem:
                return await classify_w_prob_async(
                    async_client,
                    self.model,
                    INSTRUCTION.format(target=target),
                    text,
                )

        # Filter first so we know total for tqdm
        coros = [handle_task(t) for t in tasks]
        results = []

        # Progress bar over completion of coroutines
        for coro in asyncio.as_completed(coros):  # , total=len(coros)):
            r = await coro
            results.append(r)

        return results

    def batched_analysis(
        self, tasks: list[tuple[str, str]], batch_size: int = 8
    ) -> list[ResultE[AnalysisOutput]]:
        """Run async zero-shot analysis in batches using a single event loop."""

        async def _run_batches() -> list[AnalysisOutput]:
            # Use one async client for the whole run to avoid re-inits.
            async_client = self.client.aio
            batch_outputs: list[AnalysisOutput] = []
            try:
                for start in tqdm(range(0, len(tasks), batch_size)):
                    batch_tasks = tasks[start : start + batch_size]
                    batch_outputs.extend(
                        await self._run_async_analysis(batch_tasks, async_client)
                    )
                return batch_outputs
            finally:
                await async_client.aclose()

        batch_outputs = asyncio.run(_run_batches())
        return [Success(output) for output in batch_outputs]
