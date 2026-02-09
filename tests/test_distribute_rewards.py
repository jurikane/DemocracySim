import unittest
from unittest.mock import MagicMock
from tests.factory import create_test_model

class TestDistributeRewards(unittest.TestCase):
    def setUp(self):
        self.model, _ = create_test_model(num_areas=1)
        self.model.initialize_area = MagicMock()

    def test_distribute(self):
        area = self.model.areas[0]
        for agent in area.agents:
            agent.update_known_cells(area)
        area.conduct_election()  # Ensure there's a result
        area._distribute_rewards()
        for agent in area.agents:
            self.assertGreaterEqual(agent.assets, 0)
