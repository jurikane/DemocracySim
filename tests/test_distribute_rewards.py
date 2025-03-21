import unittest
from unittest.mock import MagicMock
from tests.factory import create_default_model

class TestDistributeRewards(unittest.TestCase):
    def setUp(self):
        self.model = create_default_model(num_areas=1)
        self.model.initialize_area = MagicMock()

    def test_distribute(self):
        area = self.model.areas[0]
        area._conduct_election()  # Ensure there's a result
        area._distribute_rewards()
        for agent in area.agents:
            self.assertGreaterEqual(agent.assets, 0)
