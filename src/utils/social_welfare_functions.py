"""
Representation contract:
- Input: ScoreVector table (pref_table)
  * rows = agents
  * columns = options
  * values = disagreement/oppose scores (lower = better)
- Output: Ordering (permutation) with best option first.

This design allows non-discrete and non-equidistant preferences.

Implemented rules (schema B1):
- majority_rule (first-choice plurality after tie-prep)
- approval_voting (canonical fixed-threshold approval mapping)
- approval_voting_custom (legacy adaptive approval mapping; non-canonical)
- utilitarian_rule (minimize total disagreement)
- borda_rule (positional scoring derived from per-voter orderings)
"""

from __future__ import annotations

import numpy as np

from src.utils.representations import validate_ordering, scores_to_ordering


# Canonical approval mapping threshold on normalized disagreement scores in [0,1].
# Lower score = better; approve if score <= tau.
APPROVAL_THRESHOLD_TAU = 0.5


def complete_ranking(
    ordering: np.ndarray,
    num_options: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    This function adds options that are not in the ordering in a random order.

    Args:
        ordering (nd.ndarray): Partial ordering (permutation) of option indices.
        num_options (int): The total number of options.
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Completed ordering of length `num_options`.
    """
    all_options = np.arange(num_options)
    mask = np.isin(all_options, ordering, invert=True)
    non_included_options = all_options[mask]
    rng.shuffle(non_included_options)
    return np.concatenate((ordering, non_included_options))

def run_tie_breaking_preparation_for_majority(
    pref_table: np.ndarray,
    eps: float = 1e-9,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Prepare ballots for majority rule by breaking *only first-choice ties*.

    Args:
        pref_table (np.ndarray): Preferences per agent (rows) per option (cols).
        eps (float): Tiny jitter amplitude used only on tied minimal entries.
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Table where per-row minimal-score ties are broken.
    """
    prepared = np.array(pref_table, dtype=np.float64, copy=True)
    if prepared.ndim != 2:
        raise ValueError("pref_table must be 2D")
    n, m = prepared.shape
    if m <= 0:
        return prepared

    for i in range(n):
        row = prepared[i]
        min_val = float(np.min(row))
        # Tie only among exactly equal first-choice values.
        tied_min = np.flatnonzero(row == min_val)
        if tied_min.size > 1:
            jitter = np.zeros(m, dtype=np.float64)
            jitter[tied_min] = rng.uniform(-eps, eps, size=tied_min.size)
            prepared[i] = row + jitter
    return prepared

def majority_rule(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """
    This function implements the majority rule social welfare function.

    Args:
        pref_table (np.ndarray): ScoreVector table (disagreement values)
            per agent (rows) per option (cols), lower = better.
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Ordering (permutation) of options.
    """
    if pref_table.ndim != 2:
        raise ValueError("pref_table must be 2D")
    n, m = pref_table.shape  # n agents, m options
    if m <= 0:
        return np.asarray([], dtype=np.int64)
    if n <= 0:
        return np.arange(m, dtype=np.int64)

    prepared = run_tie_breaking_preparation_for_majority(pref_table, rng=rng)
    first_choices = np.argmin(prepared, axis=1).astype(np.int64)
    # Preserve legacy majority tie resolution pattern:
    # randomize ballot order, then stable-sort by plurality counts.
    rng.shuffle(first_choices)
    first_choice_counts: dict[int, int] = {}
    for choice in first_choices:
        c = int(choice)
        first_choice_counts[c] = first_choice_counts.get(c, 0) + 1
    option_count_pairs = list(first_choice_counts.items())
    option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    ordering = np.array([pair[0] for pair in option_count_pairs], dtype=np.int64)
    if ordering.shape[0] < m:
        ordering = complete_ranking(ordering, m, rng=rng)
    validate_ordering(ordering, m)
    return ordering

def preprocessing_for_approval(
    pref_table: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Interpret values below threshold as approval.

    This function prepares the preference table for approval voting
    by interpreting every value below a threshold as an approval.
    Beware: the values are distance/disagreement => smaller = less disagreement
    The standard threshold is 1/m (m = number of options).
    The reasoning is that if the preferences are normalized,
    1/m ensures the threshold to be proportionate to the number of options.
    It also ensures that, on average, half of the options will be approved.
    The actual number of approved options, however,
    can still vary depending on the specific values in the preference table.

    Args:
        pref_table (np.ndarray): Preferences table.
        threshold (float | None): Approval threshold; defaults to 1/m.

    Returns:
        np.ndarray: Binary approvals with shape of `pref_table`.
    """
    if threshold is None:
        threshold = 1 / pref_table.shape[1]
    return (pref_table < threshold).astype(int)


def imp_prepr_for_approval(pref_table: np.ndarray) -> np.ndarray:
    """
    This is just like preprocessing_for_approval, but more intelligent.
    It sets the threshold depending on the variances.

    Args:
        pref_table (np.ndarray): Preferences table.

    Returns:
        np.ndarray: Binary approvals with shape of `pref_table`.
    """
    # The threshold is set according to the variances
    threshold = np.mean(pref_table, axis=1) - np.var(pref_table, axis=1)
    return (pref_table < threshold.reshape(-1, 1)).astype(int)


def approval_voting_custom(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """
    Legacy/custom approval mapping using adaptive per-voter threshold (mean-variance).

    Kept for exploratory comparisons; not used as canonical approval in thesis baseline.
    """
    pref_table = imp_prepr_for_approval(pref_table)
    approval_counts = np.sum(pref_table, axis=0)
    eps = 1e-6
    noise = rng.uniform(-eps, eps, len(approval_counts))
    return np.argsort(-(approval_counts + noise))


def approval_voting(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """
    Canonical approval voting with fixed threshold mapping.

    Args:
        pref_table (np.ndarray): ScoreVector table (disagreement values).
            per agent (rows) per option (cols), lower = better.
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Ordering (permutation) of options.
    """
    if pref_table.ndim != 2:
        raise ValueError("pref_table must be 2D")
    _n, m = pref_table.shape
    if m <= 0:
        return np.asarray([], dtype=np.int64)

    approvals = preprocessing_for_approval(pref_table, threshold=APPROVAL_THRESHOLD_TAU)
    approval_counts = np.sum(approvals, axis=0).astype(np.int64)
    # Tie-break policy:
    # 1) higher approval count wins
    # 2) lower total disagreement wins (content-based, label-neutral)
    # 3) randomized final tie-break (deterministic for fixed seed)
    totals = np.sum(pref_table, axis=0).astype(np.float64)
    rand = rng.random(m)
    ordering = np.lexsort((rand, totals, -approval_counts)).astype(np.int64)
    validate_ordering(ordering, m)
    return ordering


def utilitarian_rule(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Utilitarian (score) rule on disagreement scores.

    Semantics:
    - pref_table entries are disagreement/oppose scores in [0,1], lower=better.
    - Aggregate by minimizing total disagreement across voters:
        total[j] = sum_i pref_table[i, j]
      Lower total => better.

    Tie-breaking:
    - Random but deterministic given `rng`, using a secondary random key.
    """
    if pref_table.ndim != 2:
        raise ValueError("pref_table must be 2D")
    n, m = pref_table.shape
    if m <= 0:
        return np.asarray([], dtype=np.int64)
    totals = np.sum(pref_table, axis=0).astype(np.float64)
    # Primary key totals (ascending), secondary random key.
    rand = rng.random(m)
    ordering = np.lexsort((rand, totals))
    validate_ordering(ordering, m)
    return ordering


def borda_rule(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Borda count derived from per-voter orderings of the ScoreVector.

    We interpret each row as a (possibly tied) preference ordering by sorting
    scores ascending (lower=better). Then assign Borda points:
      best gets m-1, next m-2, ... worst gets 0.

    Tie-breaking:
    - per-voter randomized tie-breaking using `rng` via scores_to_ordering.
    - final ordering tie-breaks by a secondary random key.
    """
    if pref_table.ndim != 2:
        raise ValueError("pref_table must be 2D")
    n, m = pref_table.shape
    if m <= 0:
        return np.asarray([], dtype=np.int64)

    points_by_rank = np.arange(m - 1, -1, -1, dtype=np.float64)  # length m
    totals = np.zeros(m, dtype=np.float64)
    for i in range(n):
        ordering = scores_to_ordering(pref_table[i], rng=rng)
        # ordering[0] is best -> gets m-1 points, etc.
        totals[ordering] += points_by_rank

    # Higher totals => better.
    rand = rng.random(m)
    ordering = np.lexsort((rand, -totals))
    validate_ordering(ordering, m)
    return ordering


def continuous_score_voting(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Continuous score voting returning an Ordering (permutation) with RNG noise.

    Args:
        pref_table (np.ndarray): ScoreVector table (disagreement values).
            per agent (rows) per option (cols), lower = better.
        rng (np.random.Generator): Random number generator.
    """
    scores = np.sum(pref_table, axis=0)
    eps = 1e-8
    noise = rng.uniform(-eps, eps, len(scores))
    ordering = np.argsort(-(scores + noise))
    return ordering
