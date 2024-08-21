import numpy as np
from democracy_sim.social_welfare_functions import majority_rule

simple = np.array([
    [0.5, 0.4, 0.1],
    [0.1, 0.5, 0.4],
    [0.4, 0.5, 0.1],
    [0.1, 0.4, 0.5],
    [0.1, 0.4, 0.5],
    [0.1, 0.4, 0.5]
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
    [0.4, 0.2, 0.3, 0.1, 0. ],
    [0.4, 0.2, 0.3, 0.1, 0. ],
    [0.4, 0.2, 0.3, 0.1, 0. ],
    [0.4, 0.2, 0.3, 0.1, 0. ],
    [0.4, 0.2, 0.3, 0.1, 0. ],
    # 4 times e,b,c,d,a
    [0. , 0.3, 0.2, 0.1, 0.4],
    [0. , 0.3, 0.2, 0.1, 0.4],
    [0. , 0.3, 0.2, 0.1, 0.4],
    [0. , 0.3, 0.2, 0.1, 0.4],
    # 3 times d,c,b,e,a
    [0. , 0.2, 0.3, 0.4, 0.1],
    [0. , 0.2, 0.3, 0.4, 0.1],
    [0. , 0.2, 0.3, 0.4, 0.1],
    # 2 times b,d,e,c,a
    [0. , 0.4, 0.1, 0.3, 0.2],
    [0. , 0.4, 0.1, 0.3, 0.2]
])  # Plurality => a, e, d, b, c ~ 0, 4, 3, 1, 2

majority_simple_cases = [
    (simple, [2, 1, 0]),
    (paradoxical, [0, 4, 3, 1, 2])
]

def random_pref_profile(num_agents, num_options):
    matrix_rand = matrix = np.random.rand(num_agents, num_options)
    # Normalize the matrix
    matrix_rand = matrix_rand / matrix_rand.sum(axis=1, keepdims=True)
    return matrix_rand

def test_majority_rule():
    # Test predefined cases
    for pref_table, expected in majority_simple_cases:
        res_ranking = majority_rule(pref_table)
        assert list(res_ranking) == expected
    winners_from_ties = {}


def test_majority_rule_with_ties_all():
    with_ties_all = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])  # all equally possible
    winners_from_ties = {}
    for _ in range(500):
        ranking = majority_rule(with_ties_all)
        winner = ranking[0]
        winners_from_ties[winner] = winners_from_ties.get(winner, 0) + 1
    winners = list(winners_from_ties.keys())
    winners.sort()
    assert winners == [0, 1, 2, 3]


def test_majority_with_ties_all_ab():
    with_ties_all_ab = np.array([
        [0.3, 0.3, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25]
    ])  # all possible (a or b up first is more likely)

    winners_from_ties = {}
    for _ in range(100):
        ranking = majority_rule(with_ties_all_ab)
        winner = ranking[0]
        winners_from_ties[winner] = winners_from_ties.get(winner, 0) + 1
    winners = list(winners_from_ties.keys())
    print(f"Winners from ties (all_ab): {winners_from_ties}")
    assert winners == [0, 1, 2, 3]
    assert winners_from_ties[0] > winners_from_ties[2]
    assert winners_from_ties[1] > winners_from_ties[3]

def test_majority_with_ties_ab():
    with_ties_ab = np.array([
        [0.3, 0.3, 0.2, 0.2],
        [0.3, 0.3, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25]
    ])  # all possible (a or b up first is more likely)
    winners_from_ties = {}
    for _ in range(100):
        ranking = majority_rule(with_ties_ab)
        winner = ranking[0]
        winners_from_ties[winner] = winners_from_ties.get(winner, 0) + 1
    winners = list(winners_from_ties.keys())
    assert winners == [0, 1]

    # Test with random matrix
    matrix_rand = random_pref_profile(5, 4)
    print(f"Random matrix: {matrix_rand}")
    result = majority_rule(matrix_rand)
    print(f"Result: {result}")
    #assert all(isinstance(x, int) for x in result)