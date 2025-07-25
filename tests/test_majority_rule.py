import numpy as np
import time
from democracy_sim.social_welfare_functions import majority_rule

# Simple and standard cases (lower values = higher rank)

simple = np.array([
    [0.1, 0.4, 0.5],
    [0.5, 0.1, 0.4],
    [0.4, 0.1, 0.5],
    [0.5, 0.4, 0.1],
    [0.5, 0.4, 0.1],
    [0.5, 0.4, 0.1]
])  # => c, b, a ~ 2, 1, 0

# Following "paradoxical" example is taken from
# https://pub.dss.in.tum.de/brandt-research/minpara.pdf
#
#    5 4 3 2
#    -------
#    a e d b
#    c b c d
#    b c b e
#    d d e c
#    e a a a

paradoxical = np.array([
    # 5 times a,c,b,d,e --> 0.4, 0.2, 0.3, 0.1, 0.
    # 5 times a,c,b,d,e --> 0., 0.2, 0.1, 0.3, 0.4
    [0. , 0.2, 0.1, 0.3, 0.4],
    [0. , 0.2, 0.1, 0.3, 0.4],
    [0. , 0.2, 0.1, 0.3, 0.4],
    [0. , 0.2, 0.1, 0.3, 0.4],
    [0. , 0.2, 0.1, 0.3, 0.4],
    # 4 times e,b,c,d,a
    [0.4, 0.1, 0.2, 0.3, 0. ],
    [0.4, 0.1, 0.2, 0.3, 0. ],
    [0.4, 0.1, 0.2, 0.3, 0. ],
    [0.4, 0.1, 0.2, 0.3, 0. ],
    # 3 times d,c,b,e,a
    [0.4, 0.2, 0.1, 0. , 0.3],
    [0.4, 0.2, 0.1, 0. , 0.3],
    [0.4, 0.2, 0.1, 0. , 0.3],
    # 2 times b,d,e,c,a
    [0.4, 0. , 0.3, 0.1, 0.2],
    [0.4, 0. , 0.3, 0.1, 0.2]
])  # Plurality => a, e, d, b, c ~ 0, 4, 3, 1, 2

majority_simple_cases = [
    (simple, [2, 1, 0]),
    (paradoxical, [0, 4, 3, 1, 2])
]

def test_majority_rule():
    # Test predefined cases
    for pref_table, expected in majority_simple_cases:
        res_ranking = majority_rule(pref_table)
        assert list(res_ranking) == expected

def majority_rule_with_ties_all(pref_rel, expected_winners, iterations=1000):
    """
    Run majority rule with ties multiple times, check winners
    and calculate the coefficient of variation (CV) of the winners.
    :param pref_rel: Preference relation matrix.
    :param expected_winners: An ordered list of expected winners, i.e. [0, 1].
    :param iterations: Number of iterations.
    -------
    :return: Coefficient of variation (CV) of the winners.
    """
    winners_from_ties = {}
    for _ in range(iterations):
        ranking = majority_rule(pref_rel)
        winner = ranking[0]
        winners_from_ties[winner] = winners_from_ties.get(winner, 0) + 1
    winners = list(winners_from_ties.keys())
    winners.sort()
    assert winners == expected_winners
    counts = np.array(list(winners_from_ties.values()))
    # Calculate the coefficient of variation (CV)
    cv = np.std(counts) / np.mean(counts)
    return cv

# Cases with ties - "all equally possible"

with_ties_all = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])

with_overall_tie = np.array([
    [0.4, 0.3, 0.2, 0.1],
    [0.1, 0.4, 0.3, 0.2],
    [0.2, 0.1, 0.4, 0.3],
    [0.3, 0.2, 0.1, 0.4],
])

with_ties_mixed = np.array([
    [0.4, 0.3, 0.2, 0.1],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.1, 0.4, 0.3, 0.2],
    [0.2, 0.1, 0.4, 0.3],
    [0.25, 0.25, 0.25, 0.25],
    [0.3, 0.2, 0.1, 0.4],
])

all_equally_possible = [with_ties_all, with_overall_tie, with_ties_mixed]

def test_equally_possible(cv_threshold=0.125):
    for pref_rel in all_equally_possible:
        cv = majority_rule_with_ties_all(pref_rel, [0, 1, 2, 3])
        print(f"CV: {cv}")
        assert cv < cv_threshold

# Cases with ties - "not all equally possible"
with_ties_unequal = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.3, 0.2, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])

with_ties_all_ab = np.array([
        [0.3, 0.3, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25]
    ])  # all possible (a or b up first is more likely)

with_ties_ab = np.array([
        [0.3, 0.3, 0.2, 0.2],
        [0.3, 0.3, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25]
    ])  # all possible (a or b up first is more likely)

with_ties_unequal = [with_ties_unequal, with_ties_all_ab, with_ties_ab]

def test_with_ties_unequal():
    for pref_rel in with_ties_unequal:
        cv = majority_rule_with_ties_all(pref_rel, [0, 1, 2, 3])
        print(f"CV: {cv}")
        assert cv > 0.125

# Random matrix

def random_pref_profile(num_agents, num_options):
    rand_matrix = np.random.rand(num_agents, num_options)
    # Normalize the matrix
    matrix_rand = rand_matrix / rand_matrix.sum(axis=1, keepdims=True)
    return matrix_rand

def majority_rule_with_rand_matrix(num_agents, num_options, iterations=1000):
    """
    Run majority rule with ties multiple times, check winners
    and calculate the coefficient of variation (CV) of the winners.
    :param num_agents: Number of agents.
    :param num_options: Number of options.
    :param iterations: Number of iterations.
    -------
    :return: Dictionary of winner counts {option: count}.
    """
    winner_counts = {}
    for _ in range(iterations):
        # Create random matrix
        matrix_rand = random_pref_profile(num_agents, num_options)
        ranking = majority_rule(matrix_rand)
        winner = ranking[0]
        # Count winners
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
    return winner_counts


def test_with_random_matrix_small():
    """
    Test majority rule on a small random matrix with many iterations.
    """
    num_agents = np.random.randint(2, 200)
    # Keep num options small to expect all options to win at least once.
    num_options = np.random.randint(2, 90)
    iterations = 100*num_options
    start_time = time.time()
    wc = majority_rule_with_rand_matrix(num_agents, num_options, iterations)
    stop_time = time.time()
    # Extract winners from winner-counts dictionary and sort them
    sorted_winners = list(wc.keys())
    sorted_winners.sort()
    assert sorted_winners == list(range(num_options))
    # Extract count values
    counts = np.array(list(wc.values()))
    # Calculate the coefficient of variation (CV)
    cv = np.std(counts) / np.mean(counts)
    assert cv < 0.15
    print(f"\nCV: {cv}")
    # Print the time taken
    elapsed_time = stop_time - start_time
    print(f"\nTime taken: {elapsed_time:.2f} sec. On {iterations} iterations."
          f"With {num_agents} agents and {num_options} options.")


def test_with_random_matrix_large():
    """
    Test majority rule on a large random matrix (many agents, many options).
    """
    num_its = 100
    num_agents = np.random.randint(1000, 3000)
    num_options = np.random.randint(2000, 3000)
    # Run majority rule test with random matrix
    start_time = time.time()
    wc = majority_rule_with_rand_matrix(num_agents, num_options, num_its)
    stop_time = time.time()
    # Len of winners should be approximately equal to the number of iterations
    # because with a large number of options, winners should be mostly unique.
    winners, counts = list(wc.keys()), list(wc.values())
    assert abs(np.mean(counts) - 1) < 0.1
    assert abs((len(winners) / num_its) - 1) < 0.1
    # Calculate the coefficient of variation (CV)
    cv = np.std(counts) / np.mean(counts)
    assert cv < 0.2
    # Print the time taken
    elapsed_time = stop_time - start_time
    print(f"\nTime taken: {elapsed_time:.2f} sec. On {num_its} iterations."
          f"With {num_agents} agents and {num_options} options.")
