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
    :param pref_table: The agent's preferences as a NumPy matrix
            containing the normalized ranking vectors of all agents.
    :return: The resulting preference ranking
    """
    # Count how often each ordering appears
    first_choices = np.argmax(pref_table, axis=0)
    first_choice_counts = {}
    for choice in first_choices:
        first_choice_counts[choice] = first_choice_counts.get(choice, 0) + 1
    option_count_pairs = list(first_choice_counts.items())
    option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in option_count_pairs]


def majority_rule_2(pref_table):
    """
    This function implements the majority rule social welfare function.
    :param pref_table: The agent's preferences as a NumPy matrix
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


def approval_voting(pref_table):
    # TODO !!!!!!!!!!!!!!!!!
    # => How should pref_tables be like / how do they need to be like (options=colors or options=combinations)???
    # !!!!!!!!!!!!!!!
    # ANSWER:
    # they have to span the options - not the colors
    """
    This function implements the approval voting social welfare function.
    :param pref_table: The agent's preferences as a NumPy matrix
            containing the normalized ranking vectors of all agents.
    :return: The resulting preference ranking
    """
    # # Count how often each option is approved
    # approval_counts = np.sum(pref_table, axis=0)
    # option_count_pairs = list(enumerate(approval_counts))
    # option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    # return [pair[0] for pair in option_count_pairs]
    pass
