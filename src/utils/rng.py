import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RNGManager:
    seed: Optional[int] = None
    np_main: np.random.Generator = field(default_factory=np.random.default_rng)
    np_viz: np.random.Generator = field(default_factory=np.random.default_rng)
    np_debug: np.random.Generator = field(default_factory=np.random.default_rng)
    np_participation: np.random.Generator = field(default_factory=np.random.default_rng)
    np_voting: np.random.Generator = field(default_factory=np.random.default_rng)
    np_puzzle: np.random.Generator = field(default_factory=np.random.default_rng)
    py: random.Random = field(default_factory=random.Random)
    py_viz: random.Random = field(default_factory=random.Random)
    py_debug: random.Random = field(default_factory=random.Random)
    py_participation: random.Random = field(default_factory=random.Random)
    py_voting: random.Random = field(default_factory=random.Random)
    py_puzzle: random.Random = field(default_factory=random.Random)

    def set_seed(self, seed: Optional[int]) -> None:
        self.seed = seed
        if seed is None:
            ss = np.random.SeedSequence()
        else:
            ss = np.random.SeedSequence(int(seed))
        ss_main, ss_viz, ss_debug, ss_participation, ss_voting, ss_puzzle = ss.spawn(6)

        self.np_main = np.random.default_rng(ss_main)
        self.np_viz = np.random.default_rng(ss_viz)
        self.np_debug = np.random.default_rng(ss_debug)
        self.np_participation = np.random.default_rng(ss_participation)
        self.np_voting = np.random.default_rng(ss_voting)
        self.np_puzzle = np.random.default_rng(ss_puzzle)

        self.py = random.Random(int(ss_main.generate_state(1)[0]))
        self.py_viz = random.Random(int(ss_viz.generate_state(1)[0]))
        self.py_debug = random.Random(int(ss_debug.generate_state(1)[0]))
        self.py_participation = random.Random(int(ss_participation.generate_state(1)[0]))
        self.py_voting = random.Random(int(ss_voting.generate_state(1)[0]))
        self.py_puzzle = random.Random(int(ss_puzzle.generate_state(1)[0]))


RNG = RNGManager()


def set_seed(seed: Optional[int]) -> None:
    if seed is not None:  # Make user aware of the seed setting.
        print(f"Set models random seed to {seed}")
    RNG.set_seed(seed)


def np_rng() -> np.random.Generator:
    return RNG.np_main


def np_rng_viz() -> np.random.Generator:
    return RNG.np_viz


def np_rng_debug() -> np.random.Generator:
    return RNG.np_debug


def np_rng_participation() -> np.random.Generator:
    return RNG.np_participation


def np_rng_voting() -> np.random.Generator:
    return RNG.np_voting


def np_rng_puzzle() -> np.random.Generator:
    return RNG.np_puzzle


def py_rng() -> random.Random:
    return RNG.py


def py_rng_viz() -> random.Random:
    return RNG.py_viz


def py_rng_debug() -> random.Random:
    return RNG.py_debug


def py_rng_participation() -> random.Random:
    return RNG.py_participation


def py_rng_voting() -> random.Random:
    return RNG.py_voting


def py_rng_puzzle() -> random.Random:
    return RNG.py_puzzle
