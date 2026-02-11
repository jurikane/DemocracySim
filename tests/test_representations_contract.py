from __future__ import annotations

import numpy as np
import pytest

from src.utils.representations import (
    validate_ordering,
    validate_rank_vector,
    validate_distribution,
    validate_score_vector,
    ordering_to_ranks,
    ranks_to_ordering,
    scores_to_ordering,
    distribution_to_ordering,
)


def test_ordering_rank_round_trip() -> None:
    ordering = np.array([2, 0, 1], dtype=np.int64)
    ranks = ordering_to_ranks(ordering)
    back = ranks_to_ordering(ranks)
    assert np.array_equal(ordering, back)


def test_scores_and_distribution_to_ordering() -> None:
    scores = np.array([0.2, 0.9, 0.1], dtype=np.float64)
    ordering = scores_to_ordering(scores)
    assert np.array_equal(ordering, np.array([2, 0, 1]))

    dist = np.array([0.1, 0.7, 0.2], dtype=np.float64)
    ordering = distribution_to_ordering(dist)
    assert np.array_equal(ordering, np.array([1, 2, 0]))


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        validate_ordering(np.array([0, 0, 1]))

    with pytest.raises(ValueError):
        validate_rank_vector(np.array([0, -1, 2]))

    with pytest.raises(ValueError):
        validate_distribution(np.array([0.2, -0.1, 0.9]))

    with pytest.raises(ValueError):
        validate_score_vector(np.array([0.0, np.nan, 1.0]))
