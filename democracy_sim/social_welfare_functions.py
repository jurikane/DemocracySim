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


def complete_ranking(ranking: np.array, num_options: int):
    """
    This function adds options that are not in the ranking in a random order.
    :param ranking: The ranking to be completed with the missing options.
    :param num_options: The total number of options.
    -------
    :return: The completed ranking.
    """
    all_options = np.arange(num_options)
    mask = np.isin(all_options, ranking, invert=True)
    non_included_options = all_options[mask]
    np.random.shuffle(non_included_options)
    return np.concatenate((ranking, non_included_options))


def majority_rule(pref_table, noise_factor=100):
    """
    This function implements the majority rule social welfare function.
    :param pref_table: The agent's preferences as a NumPy matrix
            containing the normalized ranking vectors of all agents.
    :param noise_factor: Influences the amount of noise to be added
           to the preference table to break ties (higher is less noise).
    :return: The resulting preference ranking (beware: its not a pref. relation)
    """
    n, m = pref_table.shape  # n agents, m options
    # Add a tiny amount of random noise to break ties
    # Beware: Even with noise added, argmax will have a bias for lower indices
    # on arrays with too many ties. So this is just a small tiebreaker.
    variances = np.var(pref_table, axis=0) + 1e-10
    # Generate noise based on the variances to sensitively break ties
    noise_eps = variances / noise_factor
    noise = np.random.uniform(-noise_eps, noise_eps, (n, m))
    pref_table += noise
    # Count how often an option is ranked first (indexes of max values)
    first_choices = np.argmax(pref_table, axis=1)
    # To avoid a bias toward voters with low indices in the counting, we shuffle
    np.random.shuffle(first_choices)
    first_choice_counts = {}
    for choice in first_choices:
        first_choice_counts[choice] = first_choice_counts.get(choice, 0) + 1
    # Get the ranking from the counts
    option_count_pairs = list(first_choice_counts.items())
    option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    ranking = np.array([pair[0] for pair in option_count_pairs])
    # Faster:
    # count_pairs = np.array(option_count_pairs)
    # # Sort the array by the second element in descending order
    # sorted_indices = np.argsort(count_pairs[:, 1])[::-1]
    # count_pairs = count_pairs[sorted_indices]
    # ranking = count_pairs[:, 0].astype(int)
    # Fill up the ranking with the missing options (if any)
    if ranking.shape[0] < m:
        ranking = complete_ranking(ranking, m)
    return ranking


#def majority_rule_2(pref_table):
    # """
    # This function implements the majority rule social welfare function.
    # :param pref_table: The agent's preferences as a NumPy matrix
    #         one row one agent, column number is color,
    #         values are the guessed distribution values (not the ranking!)
    # :return: The resulting preference ranking
    # """

    # # Count how often each color is guessed as
    # first_choices = np.argmax(pref_table, axis=0)
    # first_choice_counts = {}
    # for choice in first_choices:
    #     first_choice_counts[choice] = first_choice_counts.get(choice, 0) + 1
    # option_count_pairs = list(first_choice_counts.items())
    # option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    # return [pair[0] for pair in option_count_pairs]


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
