"""
Representation contract:
- Input: ScoreVector table (pref_table)
  * rows = agents
  * columns = options
  * values = disagreement/oppose scores (lower = better)
- Output: Ordering (permutation) with best option first.

This design allows non-discrete and non-equidistant preferences.
"""

from __future__ import annotations

import numpy as np


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
    noise_factor: int = 100,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    This function prepares the preference table for majority rule such that
    it handles ties in the voters' preferences.
    Because majority rule cannot usually deal with ties.
    The tie breaking is randomized to ensure anonymity and neutrality.

    Args:
        pref_table (np.ndarray): Preferences per agent (rows) per option (cols).
        noise_factor (int): Controls noise magnitude.
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Table without ties in first choices.
    """
    # Add some random noise to break ties (based on the variances)
    variances = np.var(pref_table, axis=1)
    # If variances are zero, all values are equal, then select a random option
    mask = (variances == 0)
    # Split
    pref_tab_var_zero = pref_table[mask]
    pref_tab_var_non_zero = pref_table[~mask]
    n, m = pref_tab_var_non_zero.shape

    # Set exactly one option to 0 (the first choice) and the rest to 1/(m-1)
    pref_tab_var_zero.fill(1 / (m - 1))
    for i in range(pref_tab_var_zero.shape[0]):
        rand_option = int(rng.integers(0, m))
        pref_tab_var_zero[i, rand_option] = 0
    # On the non-zero part, add some noise to the values to break ties
    non_zero_variances = variances[~mask]
    # Generate noise based on the variances
    noise_eps = non_zero_variances / noise_factor
    noise = rng.uniform(-noise_eps[:, np.newaxis], noise_eps[:, np.newaxis], (n, m))
    pref_tab_var_non_zero += noise

    # Put the parts back together
    return np.concatenate((pref_tab_var_non_zero, pref_tab_var_zero))

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
    n, m = pref_table.shape  # n agents, m options
    pref_table = run_tie_breaking_preparation_for_majority(pref_table, rng=rng)
    first_choices = np.argmin(pref_table, axis=1)
    rng.shuffle(first_choices)
    first_choice_counts = {}
    for choice in first_choices:
        first_choice_counts[int(choice)] = first_choice_counts.get(int(choice), 0) + 1
    option_count_pairs = list(first_choice_counts.items())
    option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    ordering = np.array([pair[0] for pair in option_count_pairs])
    if ordering.shape[0] < m:
        ordering = complete_ranking(ordering, m, rng=rng)
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


def approval_voting(pref_table: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """
    This function implements the approval voting social welfare function.

    Args:
        pref_table (np.ndarray): ScoreVector table (disagreement values).
            per agent (rows) per option (cols), lower = better.
        rng (np.random.Generator): Random number generator.
    Returns:
        np.ndarray: Ordering (permutation) of options.
    """
    pref_table = imp_prepr_for_approval(pref_table)
    approval_counts = np.sum(pref_table, axis=0)
    eps = 1e-6
    noise = rng.uniform(-eps, eps, len(approval_counts))
    return np.argsort(-(approval_counts + noise))


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
