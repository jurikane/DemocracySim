from typing import TYPE_CHECKING, cast
import mesa
from mesa import Agent

if TYPE_CHECKING:
    from participation_model import ParticipationModel


class VoteAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model: mesa.Model):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's variable and set the initial values.
        self.wealth = 1

    def step(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.agent_scheduler.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1

    def move(self):
        if TYPE_CHECKING:  # Type hint for IDEs
            self.model = cast(ParticipationModel, self.model)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,  # Moore vs. von neumann
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


class ColorCell(mesa.Agent):
    """
    Represents a cell's color
    """

    def __init__(self, pos, model, initial_color: int, num_colors: int):
        """
        Create a cell, in the given state, at the given row, col position.
        """
        super().__init__(pos, model)
        self._row = pos[0]
        self._col = pos[1]
        self._color = initial_color
        self._next_color = None
        self.color_ids = list(range(num_colors))  # Colors as integers

    @property
    def col(self):
        """Return the col location of this cell."""
        return self._col

    @property
    def row(self):
        """Return the row location of this cell."""
        return self._row

    @property
    def color(self):
        """Return the current color of this cell."""
        return self._color

    def color_step(self):
        """
        Determines the cells' color for the next step
        """
        # _neighbor_iter = self.model.grid.iter_neighbors((self._row, self._col), True)
        # neighbors_opinion = Counter(n.get_state() for n in _neighbor_iter)
        # # Following is a a tuple (attribute, occurrences)
        # polled_opinions = neighbors_opinion.most_common()
        # tied_opinions = []
        # for neighbor in polled_opinions:
        #     if neighbor[1] == polled_opinions[0][1]:
        #         tied_opinions.append(neighbor)
        #
        # self._next_color = self.random.choice(tied_opinions)[0]
        pass

    def advance(self):
        """
        Set the state of the agent to the next state
        """
        # self._color = self._next_color
        pass
