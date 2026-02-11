from __future__ import annotations

from typing import Literal
import numpy as np
from src.utils.rng import np_rng


def validate_ordering(x: np.ndarray, n: int | None = None) -> None:
    """Validate ColorOrdering: permutation of 0...n-1 (index=rank, value=option id)."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("ordering must be 1D")
    if n is None:
        n = int(arr.size)
    if arr.size != n:
        raise ValueError("ordering length does not match n")
    expected = np.arange(n)
    if not np.array_equal(np.sort(arr), expected):
        raise ValueError("ordering must be a permutation of 0..n-1")


def validate_rank_vector(x: np.ndarray, n: int | None = None) -> None:
    """Validate RankVector: index=option id, value=rank (0 best).

    Ties are allowed; values must be integers in [0, n-1].
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("rank vector must be 1D")
    if n is None:
        n = int(arr.size)
    if arr.size != n:
        raise ValueError("rank vector length does not match n")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("rank vector must contain integers")
    if arr.min(initial=0) < 0 or arr.max(initial=0) >= n:
        raise ValueError("rank vector values must be in [0, n-1]")


def validate_score_vector(x: np.ndarray, n: int | None = None) -> None:
    """Validate ScoreVector: index=option id, real-valued score (lower=better)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("score vector must be 1D")
    if n is None:
        n = int(arr.size)
    if arr.size != n:
        raise ValueError("score vector length does not match n")
    if not np.all(np.isfinite(arr)):
        raise ValueError("score vector must be finite")


def validate_score_vector_unit_interval(x: np.ndarray, n: int | None = None, *, tol: float = 1e-6) -> None:
    """Validate ScoreVector with thesis contract bounds: values in [0,1].

    Motivation: social welfare functions assume a comparable score scale and the
    simulation should not allow out-of-range values to silently skew outcomes.
    """
    arr = np.asarray(x, dtype=np.float64)
    validate_score_vector(arr, n)
    lo = float(arr.min(initial=0.0))
    hi = float(arr.max(initial=0.0))
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError("score vector values must be in [0, 1]")


def validate_distribution(x: np.ndarray, n: int | None = None, tol: float = 1e-6) -> None:
    """Validate Distribution: index=option id, values >=0, sums to 1."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("distribution must be 1D")
    if n is None:
        n = int(arr.size)
    if arr.size != n:
        raise ValueError("distribution length does not match n")
    if float(arr.min(initial=0.0)) < -tol:
        raise ValueError("distribution must be nonnegative")
    s = float(arr.sum())
    if not np.isfinite(s) or abs(s - 1.0) > tol:
        raise ValueError("distribution must sum to 1")


def ordering_to_ranks(ordering: np.ndarray) -> np.ndarray:
    """Convert Ordering -> RankVector (rank[option]=position)."""
    arr = np.asarray(ordering, dtype=np.int64)
    validate_ordering(arr, int(arr.size))
    ranks = np.empty_like(arr)
    ranks[arr] = np.arange(arr.size)
    return ranks


def ranks_to_ordering(ranks: np.ndarray, eps = 1e-6) -> np.ndarray:
    """Convert RankVector -> Ordering via argsort with deterministic tie-breaks."""
    arr = np.asarray(ranks)
    validate_rank_vector(arr, int(arr.size))

    has_ties = len(np.unique(arr)) != len(arr)
    if has_ties:
        if not eps or eps <= 0:
            raise ValueError("Epsilon must be positive to break ties in ranks")
        noise = np_rng().uniform(-eps, eps, size=arr.size)
        return np.argsort(arr + noise, kind="stable").astype(np.int64)

    return np.argsort(arr, kind="stable").astype(np.int64)



def scores_to_ordering(
    scores: np.ndarray,
    eps = 1e-6,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert ScoreVector -> Ordering (lower score = better).
    Uses RNG noise to break ties if eps > 0, otherwise raises error on ties.
    """
    arr = np.asarray(scores, dtype=np.float64)
    validate_score_vector(arr, int(arr.size))

    if eps and eps > 0:  # Often have ties so go for tie-breaking first
        if rng is None:
            rng = np_rng()
        noise = rng.uniform(-eps, eps, size=arr.size)
        return np.argsort(arr + noise, kind="stable").astype(np.int64)
    elif len(np.unique(arr)) == len(arr): # check for ties
        return np.argsort(arr, kind="stable").astype(np.int64)
    else:
        raise ValueError("Epsilon must be positive to break ties in ranks")


def distribution_to_ordering(dist: np.ndarray, *, stable: bool = True) -> np.ndarray:
    """Convert Distribution -> Ordering (higher prob = better)."""
    arr = np.asarray(dist, dtype=np.float64)
    validate_distribution(arr, int(arr.size))
    kind = "stable" if stable else "quicksort"
    return np.argsort(arr, kind=kind)[::-1].astype(np.int64)


def scores_to_distribution(
    scores: np.ndarray,
    *,
    method: Literal["exp", "linear"] = "exp",
) -> np.ndarray:
    """Convert ScoreVector -> Distribution (lower score = higher weight).

    method:
      - "exp": weights = exp(-scores)
      - "linear": weights = max_score - score (clipped at 0)
    """
    arr = np.asarray(scores, dtype=np.float64)
    validate_score_vector(arr, int(arr.size))
    if method == "exp":
        weights = np.exp(-arr)
    elif method == "linear":
        max_score = float(np.max(arr))
        weights = np.maximum(max_score - arr, 0.0)
    else:
        raise ValueError("method must be 'exp' or 'linear'")
    s = float(weights.sum())
    if s <= 0:
        return np.full_like(weights, 1.0 / weights.size, dtype=np.float64)
    return (weights / s).astype(np.float64)
