import heapq
import random
from typing import Protocol, Literal, runtime_checkable

from returns.result import Success, Failure, Result
from ..stance_dataset import StanceDataEntry
from .utils import FuzzerErr

# (confidence, stance)
Seed = tuple[float, StanceDataEntry]

SchedulerChoice = Literal["fifo", "priority", "random", "priority_random"]


@runtime_checkable
class SeedScheduler(Protocol):
    """
    Protocol for seed schedulers
    """

    population: list[Seed]

    def pick(self) -> Result[Seed, FuzzerErr]:
        """Pick the next seed to fuzz"""
        ...

    def add(self, new_seed: Seed) -> None:
        """Add a new seed to the population"""
        ...


class FIFOScheduler:
    """FIFO seed scheduler
    uses the first-in-first-out strategy to schedule seeds.
    The oldest seed in the population is selected for fuzzing next.
    The picked seed is removed from the population.
    """

    def __init__(self) -> None:
        self.population: list[Seed] = []

    def pick(self) -> Result[Seed, FuzzerErr]:
        """Get the first seed in the population queue"""
        if not self.population:
            return Failure(FuzzerErr.EMPTY_SEED)
        return Success(self.population.pop(0))

    def add(self, new_seed: Seed) -> None:
        """Add a new seed to the end of the population queue"""
        self.population.append(new_seed)


class PriorityScheduler:
    """Priority-based seed scheduler
    maintains a min-heap of seeds ordered by confidence.
    The seed with the lower confidence score is selected for fuzzing next.
    The picked seed is removed from the population.
    """

    def __init__(self) -> None:
        self.population: list[Seed] = []

    def pick(self) -> Result[Seed, FuzzerErr]:
        """Get the seed with the highest priority (lowest confidence)"""
        if not self.population:
            return Failure(FuzzerErr.EMPTY_SEED)

        seed = heapq.heappop(self.population)
        return Success(seed)

    def add(self, new_seed: Seed) -> None:
        """Add a new seed and keep the heap ordered by confidence score"""
        heapq.heappush(self.population, new_seed)


class RandomScheduler:
    """Random seed scheduler
    randomly selects a seed from the population for fuzzing.
    All the seeds in the population have equal probability to be selected.
    The picked seed is removed from the population.
    """

    def __init__(self) -> None:
        self.population: list[Seed] = []

    def pick(self) -> Result[Seed, FuzzerErr]:
        """Randomly select a seed from the population"""
        if not self.population:
            return Failure(FuzzerErr.EMPTY_SEED)
        idx = random.randrange(len(self.population))
        return Success(self.population.pop(idx))

    def add(self, new_seed: Seed) -> None:
        """Append a newly discovered seed without reordering"""
        self.population.append(new_seed)


class PriorityRandomScheduler:
    """Priority-based random seed scheduler
    selects seeds in a priority-based random fashion from the population for fuzzing.
    Seeds are selected based on their priority and a random factor.
    The seed with lower confidence score has higher probability to be selected.
    The picked seed is removed from the population.
    """

    def __init__(self) -> None:
        self.population: list[Seed] = []

    def _weights(self) -> list[float]:
        weights: list[float] = []
        for confidence, _ in self.population:
            if confidence is None or confidence <= 0:
                weights.append(1.0)
            else:
                weights.append(1.0 / confidence)

        total = sum(weights)
        if total == 0:
            return [1.0] * len(self.population)
        # Normalizing avoids floating point blowups when confidences are tiny.
        return [w / total for w in weights]

    def pick(self) -> Result[Seed, FuzzerErr]:
        """Sample a seed using weighted priority-based randomness"""
        if not self.population:
            return Failure(FuzzerErr.EMPTY_SEED)

        weights = self._weights()
        seed = random.choices(self.population, weights=weights, k=1)[0]
        self.population.remove(seed)
        return Success(seed)

    def add(self, new_seed: Seed) -> None:
        """Append the seed before recomputing random weights"""
        self.population.append(new_seed)
