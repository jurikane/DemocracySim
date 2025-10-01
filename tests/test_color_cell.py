import unittest
from mesa import Model, space
from src.agents.color_cell import ColorCell


class DummyModel(Model):
    """Minimal model so ColorCell can be instantiated."""
    def __init__(self):
        super().__init__()
        self.grid = space.SingleGrid(10, 10, torus=True)


class TestColorCell(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()

    def test_initialization(self):
        cell = ColorCell(unique_id=1, model=self.model, pos=(2, 3), initial_color=5)

        self.assertEqual(cell.unique_id, 1)
        self.assertEqual(cell.row, 2)
        self.assertEqual(cell.col, 3)
        self.assertEqual(cell.pos, (2, 3))
        self.assertEqual(cell.color, 5)
        self.assertEqual(cell.num_agents_in_cell, 0)
        self.assertEqual(cell.agents, [])
        self.assertEqual(cell.areas, [])
        self.assertFalse(cell.is_border_cell)

    def test_add_and_remove_agents(self):
        cell = ColorCell(1, self.model, (0, 0), 1)
        agent1, agent2 = object(), object()

        cell.add_agent(agent1)
        cell.add_agent(agent2)

        self.assertEqual(cell.num_agents_in_cell, 2)
        self.assertIn(agent1, cell.agents)
        self.assertIn(agent2, cell.agents)

        cell.remove_agent(agent1)
        self.assertEqual(cell.num_agents_in_cell, 1)
        self.assertNotIn(agent1, cell.agents)

    def test_add_area(self):
        cell = ColorCell(1, self.model, (0, 0), 1)
        area = {"id": 123}

        cell.add_area(area)
        self.assertIn(area, cell.areas)

    def test_str_representation(self):
        cell = ColorCell(42, self.model, (1, 1), 7)
        s = str(cell)
        self.assertIn("Cell (42", s)
        self.assertIn("pos=(1, 1)", s)
        self.assertIn("color=7", s)
        self.assertIn("num_agents=0", s)

    # TODO: Add tests for color_step once implemented
    #  - Check that it picks a next color from neighbors
    #  - Handle ties correctly
    #
    # TODO: Add tests for advance once implemented
    #  - Ensure that color updates from _next_color


if __name__ == "__main__":
    unittest.main()