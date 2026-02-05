from math import comb
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias, Sequence

from src.utils.representations import (
    validate_ordering,
    validate_rank_vector,
    validate_score_vector,
    validate_distribution,
    ordering_to_ranks,
)

IntArray: TypeAlias = NDArray[np.int64]
FloatArray: TypeAlias = NDArray[np.float64]


def unnormalized_kendall_tau(ordering_1: IntArray, ordering_2: IntArray,
                             search_pairs: Sequence[tuple[int, int]]) -> int:
    """
    This function calculates the kendal tau distance on two orderings.
    An ordering holds the option names in the order of their rank (rank=index).

    Args:
        ordering_1 (IntArray): First Array containing ranked options.
        ordering_2 (IntArray): The second ordering array.
        search_pairs (Sequence[tuple[int,int]]): Index pairs (for efficiency).

    Returns:
        The kendall tau distance
    """
    # Rename the elements to reduce the problem to counting inversions
    mapping = {option: idx for idx, option in enumerate(ordering_1)}
    renamed_arr_2 = np.array([mapping[option] for option in ordering_2])
    # Count inversions using precomputed pairs
    kendall_distance = 0
    for i, j in search_pairs:
        if renamed_arr_2[i] > renamed_arr_2[j]:
            kendall_distance += 1
    return kendall_distance


def kendall_tau_ordering(ordering_1: IntArray, ordering_2: IntArray,
                         search_pairs: Sequence[tuple[int, int]]) -> float:
    """Normalized Kendall tau distance on orderings (permutations).

    Ordering: index = rank, value = option id.

    Args:
        ordering_1 (IntArray): First (NumPy) array containing ranked options.
        ordering_2 (IntArray): The second ordering array.
        search_pairs (Sequence[tuple[int,int]]): Index pairs.

    Returns:
        int: Normalized kendall tau distance
    """
    # The Kendall tau rank distance is a metric that counts the number
    #     of pairwise disagreements between two ranking lists.
    #     The larger the distance, the more dissimilar the two lists are.
    #     Kendall tau distance is also called bubble-sort distance.
    ordering_1 = np.asarray(ordering_1, dtype=np.int64)
    ordering_2 = np.asarray(ordering_2, dtype=np.int64)
    n = ordering_1.size
    if n > 0:
        validate_ordering(ordering_1, n)
        validate_ordering(ordering_2, n)
    dist = unnormalized_kendall_tau(ordering_1, ordering_2, search_pairs)
    max_distance = comb(n, 2)
    return dist / max_distance if max_distance > 0 else 0.0


# Backward-compatible name (ordering-based)
def kendall_tau(ordering_1: IntArray, ordering_2: IntArray,
                search_pairs: Sequence[tuple[int, int]]) -> float:
    return kendall_tau_ordering(ordering_1, ordering_2, search_pairs)


def spearman_footrule_ranks(rank_arr_1: FloatArray, rank_arr_2: FloatArray) -> float:
    """Normalized Spearman footrule on rank vectors.

    Rank vector: index = option id, value = rank (0 best).
    """
    rank_arr_1 = np.asarray(rank_arr_1)
    rank_arr_2 = np.asarray(rank_arr_2)
    n = int(rank_arr_1.size)
    if n > 0:
        validate_rank_vector(rank_arr_1, n)
        validate_rank_vector(rank_arr_2, n)
    distance = np.sum(np.abs(rank_arr_1 - rank_arr_2))
    if n % 2 == 0:
        max_dist = n**2 / 2
    else:
        max_dist = n * (n - 1) / 2
    return distance / max_dist if max_dist > 0 else 0.0


def spearman_footrule_ordering(ordering_1: IntArray, ordering_2: IntArray, _search_pairs=None) -> float:
    """Normalized Spearman footrule on orderings.

    Ordering: index = rank, value = option id.
    """
    ordering_1 = np.asarray(ordering_1, dtype=np.int64)
    ordering_2 = np.asarray(ordering_2, dtype=np.int64)
    n = ordering_1.size
    if n > 0:
        validate_ordering(ordering_1, n)
        validate_ordering(ordering_2, n)
    ranks_1 = ordering_to_ranks(ordering_1)
    ranks_2 = ordering_to_ranks(ordering_2)
    return spearman_footrule_ranks(ranks_1, ranks_2)


# Backward-compatible name (ordering-based)

def spearman(ordering_1: IntArray, ordering_2: IntArray, _search_pairs=None) -> float:
    return spearman_footrule_ordering(ordering_1, ordering_2, _search_pairs)


def l1_score_distance(score_1: FloatArray, score_2: FloatArray) -> float:
    """L1 distance between score vectors (lower=better)."""
    score_1 = np.asarray(score_1, dtype=np.float64)
    score_2 = np.asarray(score_2, dtype=np.float64)
    n = int(score_1.size)
    if n > 0:
        validate_score_vector(score_1, n)
        validate_score_vector(score_2, n)
    return float(np.sum(np.abs(score_1 - score_2)))


def distribution_distance_l1(dist_1: FloatArray, dist_2: FloatArray) -> float:
    """Normalized L1 distance between distributions (0..1).

    L1 max for distributions is 2, so normalize by 2.
    """
    dist_1 = np.asarray(dist_1, dtype=np.float64)
    dist_2 = np.asarray(dist_2, dtype=np.float64)
    n = int(dist_1.size)
    if n > 0:
        validate_distribution(dist_1, n)
        validate_distribution(dist_2, n)
    return float(np.sum(np.abs(dist_1 - dist_2)) / 2.0)


# Distance functions for rank vectors (not orderings)
# (Rank vectors hold the rank of each option (option = index).
# Not to be confused with an ordering (or sequence) where the vector
# holds options and the index is the rank.)


def kendall_tau_on_ranks(rank_arr_1: FloatArray, rank_arr_2: FloatArray,
                         search_pairs: Sequence[tuple[int, int]],
                         color_vec: IntArray) -> int:
    """
    Beware: don't use this for orderings!

    This function calculates the kendal tau distance between two rank vektors.
    (The Kendall tau rank distance is a metric that counts the number
    of pairwise disagreements between two ranking lists.
    The larger the distance, the more dissimilar the two lists are.
    Kendall tau distance is also called bubble-sort distance).
    Rank vectors hold the rank of each option (option = index).
    Not to be confused with an ordering (or sequence) where the vector
    holds options and the index is the rank.

    Args:
        rank_arr_1 (FloatArray): First Array containing the ranks of each option.
        rank_arr_2 (FloatArray): The second rank array.
        search_pairs (Sequence[tuple[int, int]]): The pairs of indices.
        color_vec: (IntArray): The vector of colors (for efficiency).

    Returns:
        int: Kendall tau distance.
    """
    # Get the ordering (option names being 0 to length)
    ordering_1 = np.argsort(rank_arr_1)
    ordering_2 = np.argsort(rank_arr_2)
    # print("Ord1:", list(ordering_1), " Ord2:", list(ordering_2))
    # Create the mapping array
    mapping_array = np.empty_like(ordering_1)  # Empty array with same shape
    mapping_array[ordering_1] = color_vec  # Fill the mapping
    # Use the mapping array to rename elements in ordering_2
    renamed_arr_2 = mapping_array[ordering_2]  # Uses NumPys advanced indexing
    # print("Ren1:",list(range(len(color_vec))), " Ren2:", list(renamed_arr_2))
    # Count inversions using precomputed pairs
    kendall_distance = 0
    # inversions = []
    for i, j in search_pairs:
        if renamed_arr_2[i] > renamed_arr_2[j]:
            # inversions.append((renamed_arr_2[i], renamed_arr_2[j]))
            kendall_distance += 1
    # print("Inversions:\n", inversions)
    return kendall_distance


def spearman_distance(rank_arr_1: FloatArray, rank_arr_2: FloatArray) -> float:
    """
    Beware: don't use this for orderings!

    This function calculates the Spearman distance between two rank vektors.
    Spearman's foot rule is a measure of the distance between ranked lists.
    It is given as the sum of the absolute differences between the ranks
    of the two lists.
    This function is meant to work with numeric values as well.
    Hence, we only assume the rank values to be comparable (e.q. normalized).

    Args:
        rank_arr_1 (FloatArray): Array containing the ranks of each option
        rank_arr_2 (FloatArray): The second rank array.

    Returns:
        float: The Spearman distance
    """
    # TODO: remove these tests (comment out) on actual simulations
    assert rank_arr_1.size == rank_arr_2.size, \
        "Rank arrays must have the same length"
    if rank_arr_1.size > 0:
        assert (rank_arr_1.min() == rank_arr_2.min()
                and rank_arr_1.max() == rank_arr_2.max()), \
            f"Error: Sequences {rank_arr_1}, {rank_arr_2} aren't comparable."
    return np.sum(np.abs(rank_arr_1 - rank_arr_2))
