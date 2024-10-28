import unittest
from democracy_sim.distance_functions import *
import numpy as np
from itertools import combinations

from democracy_sim.participation_model import ParticipationModel


class TestKendallTauDistance(unittest.TestCase):

    #TODO test normalized version

    def test_kendall_tau_on_ranks(self):

        print("TEST kendall_tau_on_ranks function")

        # Test cases kendall tau (rank-vektors)
        sequences = [
            ([1, 2, 3, 4], [1, 2, 3, 4], 0),  # Equal sequences
            ([1], [1], 0),  # Single-element sequences
            ([], [], 0),  # Empty sequences
            ([0, 3, 1, 6, 2, 5, 4], [1, 0, 3, 6, 4, 2, 5], 6),
            # Because:
            # convert to orderings =>
            # ['A','C','E','B','G','F','D'], ['B','A','F','C','E','G','D']
            # rename items s.t. first vector is sorted int vector =>
            # ['0','1','2','3','4','5','6'], ['3','0','5','1','2','4','6']
            # count inversions =>
            # (3, 0), (3, 1), (3, 2), (5, 1), (5, 2), (5, 4)
            # => 6 inversions
            # If it were an ordering instead of a rank-vektor it'd be:
            # => ['A','D','B','G','C','F','E'], ['B','A','D','G','E','C','F'],
            # => ['0','1','2','3','4','5','6'], ['2','0','1','3','6','4','5']
            # => 4 inversions: (2,0), (2,1), (6,4), (6,5) (like on wikipedia)
            ([0, 5, 2, 3, 1, 4], [5, 0, 3, 2, 4, 1], 15),
            # ordering => ['A','E','C','D','F','B'], ['B','F','D','C','E','A']
            # rename   => ['0','1','2','3','4','5'], ['5','4','3','2','1','0']
            # count  => (5, 4), (5, 3), (5, 2), (5, 1), (5, 0), (4, 3), (4, 2),
            # (4, 1), (4, 0), (3, 2), (3, 1), (3, 0), (2, 1), (2, 0), (1, 0)
            # => 15 inversions
            # ([1, 2, 3], [4, 5, 6], 0),  # No common elements
            # Again, if it were an ordering instead of a rank-vektor it'd be:
            # => ['A','F','C','D','B','E'], ['F','A','D','C','E','B'],
            # => ['0','1','2','3','4','5'], ['1','0','3','2','5','4']
            # => 3 inversions: (1,0), (3,2), (5,4)
            ([2, 3, 1], [2, 1, 3], 3),
            # C, A, B -- B, A, C
            # 3, 1, 2 (ordering but named with ints)
            # 0, 1, 2 -- 2, 1, 0
            # => inversions: (2,1), (2,0), (1,0) => 3 inversions
            ([3, 1, 2], [2, 1, 3], 1),
            # B, C, A -- B, A, C
            # 0, 1, 2 -- 0, 2, 1
            # => inversions: (2,1) => 1 inversion
            ([0.5, 1.0, 0.0], [0.5, 0.0, 1.0], 3),  # Using floats
            ([0.5, 1.0, 0.0], [0.2, 0.1, 0.8], 3),  # Using floats but not equal
            # Ties are problematic as they break the metric property here
            # see 10.1137/05063088X
            ([1, 2, 2, 3], [2, 1, 3, 2], 2),  # Testing orderings with *ties*
            # 'A'>'B'='C'>'D' - 'B'>'A'='D'>'C'
            # Ord1: [0, 1, 2, 3]  Ord2: [1, 0, 3, 2]
            # Ren1: [0, 1, 2, 3]  Ren2: [1, 0, 3, 2]
            # 2 inversions: [(1, 0), (3, 2)]
            ([2, 1, 1, 1, 3], [2, 2, 3, 3, 1], 7),  # more ties
            # 'B'='C'='D'>'A'>'E' - 'E'>'A'='B'>'C'='D'
            # Ord1: [1, 2, 3, 0, 4]  Ord2: [4, 0, 1, 2, 3]
            # Ren1: [0, 1, 2, 3, 4]  Ren2: [4, 3, 0, 1, 2]
            # 7 Inversions:
            #  [(4, 3), (4, 0), (4, 1), (4, 2), (3, 0), (3, 1), (3, 2)]
            ([3, 1, 1, 2, 2, 1, 3], [2, 2, 1, 3, 1, 1, 1], 10),  # more ties
            # 'B'='C'='F'>'D'='E'>'A'='G' - 'C'='E'='F'='G'>'A'='B'>'D'
            # Ord1: [1, 2, 5, 3, 4, 0, 6] - Ord2: [2, 4, 5, 6, 0, 1, 3]
            # Ren1: [0, 1, 2, 3, 4, 5, 6] - Ren2: [1, 4, 2, 6, 5, 0, 3]
            ([0.1, 0.2, 0.2, 0.3], [0.2, 0.01, 0.9, 0.2], 2),
            # Ties with floats
        ]

        for seq1, seq2, expected in sequences:
            print(f"# Next #\nSeq1: {seq1}, Seq2: {seq2}")
            n = len(seq1)
            assert n == len(seq2), \
                "Test failed: sequences must have the same length"
            pairs = combinations(range(0, n), 2)
            item_vec = np.arange(n)
            # assert set(np.unique(seq1)) == set(np.unique(seq2)), \
            #     "Test failed: sequences must have the same elements"
            d = kendall_tau_on_ranks(np.array(seq1), np.array(seq2),
                                     pairs, item_vec)
            print(f"Seq1: {seq1}, Seq2: {seq2}, Expected: {expected}, Got: {d}")
            assert d == expected, f"Test failed for input {seq1}, {seq2}"

    def test_kendall_tau_on_orderings(self):

        print("\nTEST kendall_tau_on_orderings (not normalized) function\n")

        # Test cases kendall tau (on orderings)
        ordering_seqs = [
            ([1, 2, 3, 4], [1, 2, 3, 4], 0),  # Equal sequences
            ([1], [1], 0),  # Single-element sequences
            ([], [], 0),  # Empty sequences
            ([0, 3, 1, 6, 2, 5, 4], [1, 0, 3, 6, 4, 2, 5], 4),
            # Because:
            # => ['A','D','B','G','C','F','E'], ['B','A','D','G','E','C','F'],
            # => ['0','1','2','3','4','5','6'], ['2','0','1','3','6','4','5']
            # => 4 inversions: (2,0), (2,1), (6,4), (6,5) (like on wikipedia)
            ([0, 5, 2, 3, 1, 4], [5, 0, 3, 2, 4, 1], 3),
            # => ['A','F','C','D','B','E'], ['F','A','D','C','E','B'],
            # => ['0','1','2','3','4','5'], ['1','0','3','2','5','4']
            # => 3 inversions: (1,0), (3,2), (5,4)
            ([2, 3, 1], [2, 1, 3], 1),
            # B, C, A -- B, A, C
            # 0, 1, 2 -- 0, 2, 1
            # => inversions: (2,1) => 1 inversion
            ([3, 1, 2], [2, 1, 3], 3),
            # C, A, B -- B, A, C
            # 0, 1, 2 -- 2, 1, 0
            # => inversions: (2,1), (2,0), (1,0) => 3 inversions
        ]

        for seq1, seq2, expected in ordering_seqs:
            print(f"# Next #\nSeq1: {seq1}, Seq2: {seq2}")
            n = len(seq1)
            assert n == len(seq2), \
                "Test failed: sequences must have the same length"
            pairs = list(combinations(range(0, n), 2))
            # Test the ordering version
            d = unnormalized_kendall_tau(np.array(seq1), np.array(seq2), pairs)
            print(f"Seq1: {seq1}, Seq2: {seq2}, Expected: {expected}, Got: {d}")
            assert d == expected, f"Test failed for input {seq1}, {seq2}"


class TestSpearmanDistance(unittest.TestCase):

    #TODO test normalized version

    def test_spearman_distance(self):

        print("\nTEST spearman_distance function\n")

        sequences = [
            ([1, 2, 3, 4], [1, 2, 3, 4], 0),  # Equal sequences
            ([1], [1], 0),  # Single-element sequences
            ([], [], 0),  # Empty sequences
            ([1, 2, 3], [3, 2, 1], 4),  # Reversed sequences
            ([1, 2, 3], [2, 3, 1], 4),  # Different ranks
            ([1, 1, 1], [1, 1, 1], 0),  # Sequences with ties
            ([1, 2, 2, 3], [2, 1, 3, 2], 4),  # Sequences with ties
            ([0.0, 0.2, 1.0], [1.0, 0.2, 0.0], 2),  # Reversed (using floats)
            ([0.5, 1.0, 0.0], [0.5, 0.0, 1.0], 2.0),  # Using floats
            ([0.5, 1.0, 0.0], [0.2, 0.0, 1.0], 2.3),  # Non-equidistant ranks
            # ([0.5, 1.0, 0.0], [0.2, 0.1, 0.8], 2.3),  # Non-normalized
            # Using floats but not equal
        ]

        for seq1, seq2, expected in sequences:
            distance = spearman_distance(np.array(seq1), np.array(seq2))
            self.assertEqual(distance, expected,
                             f"Test failed for input {seq1}, {seq2}")
