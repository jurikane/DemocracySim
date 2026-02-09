import unittest
import numpy as np
from unittest.mock import MagicMock
from tests.factory import create_test_model

class TestTallyVotes(unittest.TestCase):
    def setUp(self):
        self.model, _ = create_test_model(num_areas=1)
        self.model.initialize_area = MagicMock()

    def test_tally_votes_array(self):
        area = self.model.areas[0]
        for agent in area.agents:
            agent.update_known_cells(area)
        votes = area._tally_votes()
        self.assertIsInstance(votes, np.ndarray)

    def test_tally_votes_empty(self):
        for agent in self.model.voting_agents:
            agent.assets = 0
        area = self.model.areas[0]
        votes = area._tally_votes()
        self.assertEqual(votes.size, 0)
