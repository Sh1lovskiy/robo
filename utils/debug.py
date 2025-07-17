from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Set


@dataclass
class DebugSampler:
    """Randomly select indices to enable debug logging."""

    total: int
    n_samples: int = 3
    seed: int | None = None
    indices: Set[int] = None

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.indices = set(rng.sample(range(self.total), min(self.n_samples, self.total)))

    def should_log(self, idx: int) -> bool:
        """Return True if ``idx`` is among the sampled indices."""
        return idx in self.indices

