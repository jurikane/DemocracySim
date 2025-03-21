import unittest
from unittest.mock import MagicMock
from tests.factory import create_default_model

# TODO add more complex tests

class TestConductElection(unittest.TestCase):
    def setUp(self):
        self.model = create_default_model(num_areas=1)
        self.model.initialize_area = MagicMock()

    def test_election_returns_integer_turnout(self):
        area = self.model.areas[0]
        turnout = area._conduct_election()
        self.assertIsInstance(turnout, int)

    def test_no_participation_scenario(self):
        for agent in self.model.voting_agents:
            agent.assets = 0
        area = self.model.areas[0]
        turnout = area._conduct_election()
        self.assertEqual(turnout, 0)