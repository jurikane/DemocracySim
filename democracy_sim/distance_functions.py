from math import comb

import numpy as np


def kendall_tau_on_ranks(rank_arr_1, rank_arr_2, search_pairs, color_vec):
    """
    DON'T USE
    (don't use for orderings!)

    This function calculates the kendal tau distance between two rank vektors.
    (The Kendall tau rank distance is a metric that counts the number
    of pairwise disagreements between two ranking lists.
    The larger the distance, the more dissimilar the two lists are.
    Kendall tau distance is also called bubble-sort distance).
    Rank vectors hold the rank of each option (option = index).
    Not to be confused with an ordering (or sequence) where the vector
    holds options and the index is the rank.
    :param rank_arr_1: First (NumPy) array containing the ranks of each option
    :param rank_arr_2: The second rank array
    :param search_pairs: The pairs of indices (for efficiency)
    :param color_vec: The vector of colors (for efficiency)
    :return: The kendall tau distance
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


def unnormalized_kendall_tau(ordering_1, ordering_2, search_pairs):
    """
    This function calculates the kendal tau distance on two orderings.
    An ordering holds the option names in the order of their rank (rank=index).
    :param ordering_1: First (NumPy) array containing ranked options
    :param ordering_2: The second ordering array
    :param search_pairs: The pairs of indices (for efficiency)
    :return: The kendall tau distance
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


def kendall_tau(ordering_1, ordering_2, model):
    """
    This calculates the normalized Kendall tau distance of two orderings.
    The Kendall tau rank distance is a metric that counts the number
    of pairwise disagreements between two ranking lists.
    The larger the distance, the more dissimilar the two lists are.
    Kendall tau distance is also called bubble-sort distance.
    An ordering holds the option names in the order of their rank (rank=index).
    :param ordering_1: First (NumPy) array containing ranked options
    :param ordering_2: The second ordering array
    :param model: Containing the pairs of indices (for efficiency)
    :return: The kendall tau distance
    """
    # TODO: remove these tests (comment out) on actual simulations to speed up
    n = ordering_1.size
    if n > 0:
        expected_arr = np.arange(n)
        assert (np.array_equal(np.sort(ordering_1), expected_arr)
                and np.array_equal(np.sort(ordering_2), expected_arr)) , \
            f"Error: Sequences {ordering_1}, {ordering_2} aren't comparable."

    # Get the unnormalized Kendall tau distance
    search_pairs = model.search_pairs
    dist = unnormalized_kendall_tau(ordering_1, ordering_2, search_pairs)
    # Maximum possible Kendall tau distance
    max_distance = comb(n, 2)  # This is n choose 2, or n(n-1)/2
    # Normalize the distance
    normalized_distance = dist / max_distance

    return normalized_distance


def spearman_distance(rank_arr_1, rank_arr_2):
    """
    Beware: don't use for orderings!

    This function calculates the Spearman distance between two rank vektors.
    Spearman's foot rule is a measure of the distance between ranked lists.
    It is given as the sum of the absolute differences between the ranks
    of the two lists.
    This function is meant to work with numeric values as well.
    Hence, we only assume the rank values to be comparable (e.q. normalized).
    :param rank_arr_1: First (NumPy) array containing the ranks of each option
    :param rank_arr_2: The second rank array
    :return: The Spearman distance
    """
    # TODO: remove these tests (comment out) on actual simulations
    assert rank_arr_1.size == rank_arr_2.size, \
        "Rank arrays must have the same length"
    if rank_arr_1.size > 0:
        assert (rank_arr_1.min() == rank_arr_2.min()
                and rank_arr_1.max() == rank_arr_2.max()), \
            f"Error: Sequences {rank_arr_1}, {rank_arr_2} aren't comparable."
    return np.sum(np.abs(rank_arr_1 - rank_arr_2))


def spearman(ordering_1, ordering_2, model):
    """
    This calculates the normalized Spearman distance between two orderings.
    Spearman's foot rule is a measure of the distance between ranked lists.
    It is given as the sum of the absolute differences between the ranks
    of the two orderings (values from 0 to n-1 in any order).
    :param ordering_1: First (NumPy) array containing the ranks of each option
    :param ordering_2: The second rank array
    :param model: The mesa model
    :return: The Spearman distance
    """
    # TODO: remove these tests (comment out) on actual simulations to speed up
    n = ordering_1.size
    if n > 0:
        expected_arr = np.arange(n)
        assert (np.array_equal(np.sort(ordering_1), expected_arr)
                and np.array_equal(np.sort(ordering_2), expected_arr)) , \
            f"Error: Sequences {ordering_1}, {ordering_2} aren't comparable."
    distance = np.sum(np.abs(ordering_1 - ordering_2))
    # Normalize
    if n % 2 == 0:  # Even number of elements
        max_dist = n**2 / 2
    else: # Odd number of elements
        max_dist = n * (n - 1) / 2
    return distance / max_dist
