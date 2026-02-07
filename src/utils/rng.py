from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RNGManager:
    seed: int | None = None
    np: np.random.Generator = field(default_factory=np.random.default_rng)
    py: random.Random = field(default_factory=random.Random)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.np = np.random.default_rng(seed)
        self.py = random.Random(seed)

        # Legacy/global fallbacks: keep deterministic if someone uses np.random/random directly.
        # Prefer using RNGManager's generators instead of global state.
        random.seed(seed)
        np.random.seed(seed)


RNG = RNGManager()


def set_seed(seed: int | None) -> None:
    RNG.set_seed(seed)


def np_rng() -> np.random.Generator:
    return RNG.np


def py_rng() -> random.Random:
    return RNG.py
