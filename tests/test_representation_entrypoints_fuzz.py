from __future__ import annotations

import numpy as np
import pytest

from src.utils.representations import (
    validate_distribution,
    validate_ordering,
    validate_score_vector,
    validate_score_vector_unit_interval,
)


def test_representation_entrypoints_accept_valid_fuzzed_vectors() -> None:
    rng = np.random.default_rng(8101)
    for n in (2, 3, 4, 6, 8):
        for _ in range(25):
            ordering = rng.permutation(n)
            validate_ordering(np.asarray(ordering, dtype=np.int64), n)

            # Dirichlet yields nonnegative vectors that sum to 1.
            dist = rng.dirichlet(np.ones(n))
            validate_distribution(np.asarray(dist, dtype=np.float64), n)

            scores = rng.uniform(0.0, 1.0, size=n)
            validate_score_vector(np.asarray(scores, dtype=np.float64), n)
            validate_score_vector_unit_interval(np.asarray(scores, dtype=np.float64), n)


def test_representation_entrypoints_reject_invalid_boundary_cases() -> None:
    # Ordering invalids
    with pytest.raises(ValueError):
        validate_ordering(np.array([0, 0, 1], dtype=np.int64), 3)
    with pytest.raises(ValueError):
        validate_ordering(np.array([0, 1], dtype=np.int64), 3)
    with pytest.raises(ValueError):
        validate_ordering(np.array([[0, 1, 2]], dtype=np.int64), 3)

    # Distribution invalids
    with pytest.raises(ValueError):
        validate_distribution(np.array([0.5, 0.6, -0.1], dtype=np.float64), 3)
    with pytest.raises(ValueError):
        validate_distribution(np.array([0.2, 0.2, 0.2], dtype=np.float64), 3)
    with pytest.raises(ValueError):
        validate_distribution(np.array([0.5, np.nan, 0.5], dtype=np.float64), 3)
    with pytest.raises(ValueError):
        validate_distribution(np.array([[0.5, 0.5]], dtype=np.float64), 2)

    # Score invalids
    with pytest.raises(ValueError):
        validate_score_vector(np.array([0.1, np.nan, 0.2], dtype=np.float64), 3)
    with pytest.raises(ValueError):
        validate_score_vector(np.array([[0.1, 0.2, 0.3]], dtype=np.float64), 3)
    with pytest.raises(ValueError):
        validate_score_vector_unit_interval(np.array([0.0, 1.2, 0.5], dtype=np.float64), 3)
    with pytest.raises(ValueError):
        validate_score_vector_unit_interval(np.array([-0.1, 0.2, 0.3], dtype=np.float64), 3)
