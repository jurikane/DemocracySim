"""
Here we define the social welfare functions that can be used in the simulation.
Beware:
We assume the preference relation in the following (unconventional) way
on purpose.
pref_table: numpy matrix with one row per agent, column number is option number
            and the values (each in [0,1]) are normalized ranking values.
The purpose of this is to allow for non-discrete and non-equidistant rankings.
"""
import numpy as np


def majority_rule(pref_table):
    """
    This function implements the majority rule social welfare function.
    :param pref_table: The agents preferences as np matrix
            one row one agent, column number is color,
            values are the guessed distribution values (not the ranking!)
    :return: The resulting preference ranking
    """
    # Count how often each color is guessed as
    first_choices = np.argmax(pref_table, axis=0)
    first_choice_counts = {}
    for choice in first_choices:
        first_choice_counts[choice] = first_choice_counts.get(choice, 0) + 1
    option_count_pairs = list(first_choice_counts.items())
    option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in option_count_pairs]


# Helper functions

def rank_arr_to_ordering(rank_arr):
    """
    This function converts a rank array to an ordering array.
    Rank vectors hold the rank of each option (option = index).
    Ordering (or sequence) vectors hold options (rank = index).
    :param rank_arr: Array of numeric values unambiguously determining a ranking
    :return: The ordering determined by the rank array (options from 1 to n)
    """
    tuples = enumerate(rank_arr, start=1)  # (option, rank)
    ordering = sorted(tuples, key=lambda x: x[1])  # Sort by rank
    return ordering
