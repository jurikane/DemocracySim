from __future__ import annotations

import numpy as np
import pytest

from src.utils.ballots import ordering_from_distribution


pytestmark = pytest.mark.phase1


def test_ordering_from_distribution_requires_rng_on_ties() -> None:
    ordering_from_distribution(np.array([0.8, 0.15, 0.05], dtype=np.float32), rng=None)
    with pytest.raises(ValueError, match="pass rng"):
        ordering_from_distribution(np.array([0.5, 0.5, 0.0], dtype=np.float32), rng=None)
