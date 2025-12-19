from mesa import Agent, Model


class ColorCell(Agent):
    """
    Represents a single cell (a field in the grid) with a specific color.

    Attributes:
        color (int): The color of the cell.
    """

    def __init__(self, unique_id: int, model: Model, pos: tuple, initial_color: int):
        """
        Initializes a ColorCell, at the given row, col position.

        Args:
            unique_id (int): The unique identifier of the cell.
            model (mesa.Model): The mesa model of which the cell is part of.
            pos (Tuple[int, int]): The position of the cell in the grid.
            initial_color (int): The initial color of the cell.
        """
        super().__init__(unique_id, model)
        # self.pos will be set by the grid when we place the agent
        self._row = pos[0]
        self._col = pos[1]
        self.color = initial_color  # The cell's current color (int)
        self._next_color = None
        self.agents = []    # TODO change to using mesas AgentSet class!
        self.areas = []    # TODO change to using mesas AgentSet class!
        self.is_border_cell = False
        # Add it to the models grid
        model.grid.place_agent(self, pos)

    def __str__(self):
        return (f"Cell ({self.unique_id}, pos={self.pos}, "
                f"color={self.color}, num_agents={self.num_agents_in_cell})")

    @property
    def row(self):
        """The row location of this cell."""
        return self.pos[0]

    @property
    def col(self):
        """The col location of this cell."""
        return self.pos[1]

    @property
    def num_agents_in_cell(self):
        """The number of agents in this cell."""
        return len(self.agents)

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def add_area(self, area):
        self.areas.append(area)

    def color_step(self):
        """
        Determines the cells' color for the next step.
        TODO
        """
        # _neighbor_iter = self.model.grid.iter_neighbors(
        #     (self._row, self._col), True)
        # neighbors_opinion = Counter(n.get_state() for n in _neighbor_iter)
        # # Following is a tuple (attribute, occurrences)
        # polled_opinions = neighbors_opinion.most_common()
        # tied_opinions = []
        # for neighbor in polled_opinions:
        #     if neighbor[1] == polled_opinions[0][1]:
        #         tied_opinions.append(neighbor)
        #
        # self._next_color = self.random.choice(tied_opinions)[0]
        pass
