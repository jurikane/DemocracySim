from __future__ import annotations

from typing import Literal
import numpy as np


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
    """Convert RankVector -> Ordering.

    Fairness + determinism contract (thesis):
    - If ties exist, deterministic tie-breaking by option id is NOT acceptable for
      decision-critical code paths (it biases toward small ids).
    - Therefore: if ties exist, call sites must provide an explicit RNG for
      randomized tie-breaking (still deterministic given seed).

    This helper is intentionally strict to prevent silent bias.
    """
    arr = np.asarray(ranks)
    validate_rank_vector(arr, int(arr.size))

    has_ties = len(np.unique(arr)) != len(arr)
    if has_ties:
        # Keep backward-compat strictness, but use a clear error.
        raise ValueError("RankVector contains ties; tie-breaking requires explicit handling.")

    return np.argsort(arr, kind="stable").astype(np.int64)



def scores_to_ordering(
    scores: np.ndarray,
    eps = 1e-6,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert ScoreVector -> Ordering (lower score = better).

    Determinism contract (thesis):
    - This function must not consume the model's main RNG implicitly.
    - If there are ties, callers MUST pass rng for randomized tie-breaking.
      Stable argsort tie-breaking is biased and not acceptable for social choice.
    - If rng is provided and eps > 0, noise is added for randomized tie-breaking
      (deterministic given seed).
    """
    arr = np.asarray(scores, dtype=np.float64)
    validate_score_vector(arr, int(arr.size))

    has_ties = len(np.unique(arr)) != len(arr)
    if has_ties:
        if rng is None:
            raise ValueError("ScoreVector contains ties; pass rng for randomized tie-breaking.")
        if not eps or eps <= 0:
            raise ValueError("eps must be > 0 to break ties with RNG noise.")
        noise = rng.uniform(-eps, eps, size=arr.size)
        return np.argsort(arr + noise, kind="stable").astype(np.int64)

    # No ties: stable argsort is fine and deterministic.
    return np.argsort(arr, kind="stable").astype(np.int64)


def distribution_to_ordering(
    dist: np.ndarray,
    *,
    stable: bool = True,
    rng: np.random.Generator | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert Distribution -> Ordering (higher prob = better).

    Fairness contract:
    - If there are ties and this ordering influences decisions, pass `rng` to break ties
      randomly (still deterministic given seed).
    - If rng is None, stable argsort is used (deterministic but biased in ties).
    """
    arr = np.asarray(dist, dtype=np.float64)
    validate_distribution(arr, int(arr.size))
    has_ties = len(np.unique(arr)) != len(arr)
    if has_ties and rng is not None and eps and eps > 0:
        noise = rng.uniform(-eps, eps, size=arr.size)
        arr = arr + noise
    else:
        print("Warning: tie-breaking is biased, if not in debug/testing, always provide rng for fair tie-breaking.")
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
