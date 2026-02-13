from __future__ import annotations

import numpy as np
import pytest

from src.utils.ballots import ordering_from_distribution


pytestmark = pytest.mark.phase1


def test_ordering_from_distribution_warns_only_on_ties_without_rng(capsys: pytest.CaptureFixture[str]) -> None:
    ordering_from_distribution(np.array([0.8, 0.15, 0.05], dtype=np.float32), rng=None)
    out = capsys.readouterr().out
    assert "tie-breaking is biased" not in out

    ordering_from_distribution(np.array([0.5, 0.5, 0.0], dtype=np.float32), rng=None)
    out = capsys.readouterr().out
    assert "tie-breaking is biased" in out
