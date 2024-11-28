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

def run_tie_breaking_preparation_for_majority(pref_table, noise_factor=100):
    """
        This function prepares the preference table for majority rule such that
        it handles ties in the voters' preferences.
        Because majority rule cannot usually deal with ties.
        The tie breaking is randomized to ensure anonymity and neutrality.
        :param pref_table: The agent's preferences.
        :param noise_factor: Influences the amount of noise to be added
               to the preference table to break ties (higher is less noise).
        :return: The preference table without ties for first choices.
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
    pref_tab_var_zero.fill(1 / (m-1))
    for i in range(pref_tab_var_zero.shape[0]):
        rand_option = np.random.randint(0, m)
        pref_tab_var_zero[i, rand_option] = 0
    # On the non-zero part, add some noise to the values to break ties
    non_zero_variances = variances[~mask]
    # Generate noise based on the variances
    noise_eps = non_zero_variances / noise_factor
    noise = np.random.uniform(-noise_eps[:, np.newaxis],
                              noise_eps[:, np.newaxis], (n, m))
    # The noise_eps[:, np.newaxis] reshapes noise_eps from shape (n,) to (n, 1)
    pref_tab_var_non_zero += noise

    # Put the parts back together
    return np.concatenate((pref_tab_var_non_zero, pref_tab_var_zero))

def majority_rule(pref_table):
    """
    This function implements the majority rule social welfare function.
    Beware: Input is a preference table (values define a ranking, index=option),
            but the output is a ranking/an ordering (values represent options).
    :param pref_table: The agent's preferences (disagreement) as a NumPy matrix
            containing the normalized ranking vectors of all agents.
            The logic: lower rank = lower disagreement
    :return: The resulting preference ranking (beware: its not a pref. relation)
    """
    n, m = pref_table.shape  # n agents, m options
    # Break ties if they exist
    pref_table = run_tie_breaking_preparation_for_majority(pref_table)
    # Count how often an option is ranked first (indexes of the min values)
    first_choices = np.argmin(pref_table, axis=1)
    # To avoid bias toward voters of low indices in the counting, we shuffle
    np.random.shuffle(first_choices)  # (crucial when counting shows ties later)
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

def preprocessing_for_approval(pref_table, threshold=None):
    """
    This function prepares the preference table for approval voting
    by interpreting every value below a threshold as an approval.
    Beware: the values are distance/disagreement => smaller = less disagreement
    The standard threshold is 1/m (m = number of options).
    The reasoning is that if the preferences are normalized,
    1/m ensures the threshold to be proportionate to the number of options.
    It also ensures that, on average, half of the options will be approved.
    The actual number of approved options, however,
    can still vary depending on the specific values in the preference table.
    :param pref_table: The agent's preferences.
    :param threshold: The threshold for approval.
    :return: The preference table with the options approved or not.
    """
    if threshold is None:
        threshold = 1 / pref_table.shape[1]
    return (pref_table < threshold).astype(int)


def imp_prepr_for_approval(pref_table, threshold=None):
    """
    This is just like preprocessing_for_approval, but more intelligent.
    It sets the threshold depending on the variances.
    :param pref_table: The agent's preferences.
    Beware: the values are disagreements => smaller = less disagreement
    :param threshold: Will be overwritten (will be set according to variances).
    :return: The preference table with the options approved or not.
    """
    threshold = np.var(pref_table, axis=0)
    return (pref_table <= threshold).astype(int)


def approval_voting(pref_table):
    """ TODO: does this take the meaning of the values into account? value = dist. = disagreement !
    This function implements the approval voting social welfare function.
    Beware: Input is a preference table (values define a ranking, index=option),
            but the output is a ranking/an ordering (values represent options).
    :param pref_table: The agent's preferences (disagreement) as a NumPy matrix
            containing the normalized ranking vectors of all agents.
            The logic: lower rank = lower disagreement
    :return: The resulting preference ranking (beware: not a pref. relation).
    """
    pref_table = imp_prepr_for_approval(pref_table)
    # Count how often each option is approved
    approval_counts = np.sum(pref_table, axis=0)
    # Add noise to break ties TODO check for bias
    eps = 1e-4
    noise = np.random.uniform(-eps, eps, len(approval_counts))
    #option_count_pairs = list(enumerate(approval_counts + noise))
    #option_count_pairs.sort(key=lambda x: x[1], reverse=True)
    #return [pair[0] for pair in option_count_pairs]
    return np.argsort(-(approval_counts + noise))  # TODO: check order (ascending/descending) - np.argsort sorts ascending


def continuous_score_voting(pref_table):
    """
    TODO: integrade and test
    This function implements a continuous score voting based on disagreement.
    Beware: Input is a preference table (values define a ranking, index=option),
            but the output is a ranking/an ordering (values represent options).
    :param pref_table: The agent's preferences (disagreement) as a NumPy matrix
            containing the normalized ranking vectors of all agents.
            The logic: lower rank = lower disagreement
    :return: The resulting preference ranking (beware: not a pref. relation).
    """
    # Sum up the disagreement for each option
    scores = np.sum(pref_table, axis=0)
    # Add noise to break ties
    eps = 1e-8
    noise = np.random.uniform(-eps, eps, len(scores))
    ranking = np.argsort(-(scores + noise))
    return ranking
