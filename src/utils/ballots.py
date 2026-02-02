from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def normalize_distribution(x: np.ndarray) -> np.ndarray:
    """Return a normalized 1D distribution (nonnegative, sums to 1).

    This is intentionally strict-ish because it's core semantics for the thesis.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("distribution must be 1D")
    # allow tiny negatives from numeric noise, but do not allow substantial negatives
    if float(np.min(x)) < -1e-6:
        raise ValueError("distribution must be nonnegative")
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0:
        # fallback to uniform
        x = np.ones_like(x, dtype=np.float32)
        s = float(x.sum())
    return (x / s).astype(np.float32)


def ordering_from_distribution(dist: np.ndarray) -> np.ndarray:
    """Convert a distribution into a ColorOrdering by descending probability."""
    dist = np.asarray(dist, dtype=np.float32)
    if dist.ndim != 1:
        raise ValueError("dist must be 1D")
    return np.argsort(dist)[::-1].astype(np.int16)


def mix_distributions(*, altruism_factor: float, est_real_dist: np.ndarray, personal_opt_dist: np.ndarray) -> np.ndarray:
    """Compute target distribution used for voting.

    Semantics:
      - altruism_factor==0 -> purely self-interest (personal_opt_dist)
      - altruism_factor==1 -> purely reality-tracking (est_real_dist)

    Returns a normalized distribution.

    Note: This function is pure (no RNG).
    """
    a = float(altruism_factor)
    if not (0.0 <= a <= 1.0):
        raise ValueError("altruism_factor must be in [0,1]")

    est = np.asarray(est_real_dist, dtype=np.float32)
    personal = np.asarray(personal_opt_dist, dtype=np.float32)
    if est.shape != personal.shape:
        raise ValueError("est_real_dist and personal_opt_dist must have same shape")

    mixed = a * est + (1.0 - a) * personal
    return normalize_distribution(mixed)


def score_options_c2(
    *,
    target_ordering: np.ndarray,
    options: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray, Sequence[tuple[int, int]]], float],
    color_search_pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """C2 scoring: convert target distribution to ordering externally, then score each option ordering.

    Contract:
    - Returns float32 vector of raw distances (lower=better)
    - Must be in [0,1] if distance_func is correctly normalized
    - No normalization is performed here

    Pure function (no RNG).
    """
    target_ordering = np.asarray(target_ordering)
    options = np.asarray(options)
    if options.ndim != 2:
        raise ValueError("options must be 2D array of orderings")

    scores = np.zeros(int(options.shape[0]), dtype=np.float32)
    for i, opt in enumerate(options):
        scores[i] = float(distance_func(target_ordering, opt, color_search_pairs))
    return scores
