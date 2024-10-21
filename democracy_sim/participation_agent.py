from typing import TYPE_CHECKING, cast
import numpy as np
from mesa import Agent
if TYPE_CHECKING:  # Type hint for IDEs
    from democracy_sim.participation_model import ParticipationModel


def combine_and_normalize(arr_1: np.array, arr_2: np.array, factor: float):
    """
    Combine two arrays weighted by a factor favoring arr_1.
    The first array is to be the estimated real distribution.
    And the other is to be the personality vector of the agent.
    :param arr_1: The first array to be combined (real distribution).
    :param arr_2: The second array to be combined (personality vector).
    :param factor: The factor to weigh the two arrays.
    :return: The normalized result of the weighted linear combination.
    """
    # Ensure f is between 0 and 1 TODO: remove this on simulations to speed up
    if not (0 <= factor <= 1):
        raise ValueError("Factor f must be between 0 and 1")
    # Linear combination
    res = factor * arr_1 + (1 - factor) * arr_2
    # Normalize/scale result s. t. it resembles a distribution vector (sum=1)
    total = sum(res)
    # assert total == 1.0, f"Sum of result is {total} and not 1.0"
    return res / total


class VoteAgent(Agent):
    """An agent that has limited knowledge and resources and
    can decide to use them to participate in elections."""

    def __init__(self, unique_id, model, pos, personality, assets=1,
                 append_to_list=True):
        """ Create a new agent.
        :param unique_id: The unique identifier of the agent.
        :param model: The simulation model of which the agent is part of.
        :type model: ParticipationModel
        :param pos: The position of the agent in the grid.
        :type pos: Tuple
        :param personality: Represents the agent's preferences among colors.
        :type personality: Numpy.ndarray
        :param assets: The wealth/assets/motivation of the agent.
        :append_to_list: Whether to add the agent to the model's agent list.
        """
        # Pass the parameters to the parent class.
        super().__init__(unique_id=unique_id, model=model)
        # The "pos" variable in mesa is special, so I avoid it here
        try:
            row, col = pos
        except ValueError:
            raise ValueError("Position must be a tuple of two integers.")
        self._position = row, col
        self._assets = assets
        self._num_elections_participated = 0
        self.personality = personality
        self.known_cells = []  # ColorCell objects the agent knows (knowledge)
        # Add the agent to the models' agent list
        if append_to_list:
            model.voting_agents.append(self)

    def __str__(self):
        return (f"Agent(id={self.unique_id}, pos={self.position}, "
                f"personality={self.personality}, assets={self.assets})")

    @property
    def position(self):
        """Return the location of the agent."""
        return self._position

    @property
    def row(self):
        """Return the row location of the agent."""
        return self._position[0]

    @property
    def col(self):
        """Return the col location of the agent."""
        return self._position[1]

    @property
    def assets(self):
        """Return the assets of this agent."""
        return self._assets

    @assets.setter
    def assets(self, value):
        self._assets = value

    @assets.deleter
    def assets(self):
        del self._assets

    @property
    def num_elections_participated(self):
        return self._num_elections_participated

    @num_elections_participated.setter
    def num_elections_participated(self, value):
        self._num_elections_participated = value

    def ask_for_participation(self, area):
        """
        The agent decides
        whether to participate in the upcoming election of a given area.
        :param area: The area in which the election takes place.
        :return: True if the agent decides to participate, False otherwise
        """
        #print("Agent", self.unique_id, "decides whether to participate",
        #      "in election of area", area.unique_id)
        # TODO Implement this (is to be decided upon a learned decision tree)
        return np.random.choice([True, False])

    def decide_altruism_factor(self, area):
        """
        Uses a trained decision tree to decide on the altruism factor.
        """
        # TODO Implement this (is to be decided upon a learned decision tree)
        # This part is important - also for monitoring - save/plot a_factors
        a_factor = np.random.uniform(0.0, 1.0)
        #print(f"Agent {self.unique_id} has an altruism factor of: {a_factor}")
        return a_factor

    def compute_assumed_opt_dist(self, area):
        """
        Computes a color distribution that the agent assumes to be an optimal
        choice in any election (regardless of whether it exists as a real option
        to vote for or not). It takes "altruistic" concepts into consideration.
        :param area: The area in which the election takes place.
        :return: The assumed optimal color distribution (normalized).
        TODO add unit test for this method
        """
        # Compute the "altruism_factor" via a decision tree
        a_factor = self.decide_altruism_factor(area)  # TODO: Implement this
        # compute the preference ranking vector as a mix between the agent's
        # own preferences/personality traits and the estimated real distribution
        est_dist = self.estimate_real_distribution(area)
        ass_opt = combine_and_normalize(self.personality, est_dist, a_factor)
        return ass_opt

    def vote(self, area):
        """
        The agent votes in the election of a given area,
        i.e., she returns a preference ranking vector over all options.
        (Options are indexes, values are preference values defining the order).
        The available options are set in the model.
        :param area: The area in which the election takes place.
        """
        # TODO Implement this (is to be decided upon a learned decision tree)
        # Compute the color distribution that is assumed to be the best choice.
        # TODO est_best_dist = self.compute_assumed_opt_dist(area)
        # Make sure that r= is normalized!
        # (r.min()=0.0 and r.max()=1.0 and all vals x are within [0.0, 1.0]!)
        ##############
        if TYPE_CHECKING:  # Type hint for IDEs
            self.model = cast(ParticipationModel, self.model)
        # For TESTING, we just shuffle the option vector (ints) then normalize
        # and interpret the result as a preference vector (values=prefs)
        # (makes no sense, but it'll work for testing)
        r = np.arange(self.model.options.shape[0])
        # Shuffle the array in place
        np.random.shuffle(r)
        r = np.array(r, dtype=float)
        r /= r.sum()
        #print("Agent", self.unique_id, "voted:", r)
        return r

    def estimate_real_distribution(self, area):
        """
        The agent estimates the real color distribution in the area based on
        her own knowledge (self.known_fields).
        """
        relevant_cells = area.filter_cells(self.known_cells)
        known_colors = np.array([cell.color for cell in relevant_cells])
        unique, counts = np.unique(known_colors, return_counts=True)
        distribution = np.zeros(self.model.num_colors)
        distribution[unique] = counts / known_colors.size
        return distribution


class ColorCell(Agent):
    """
    Represents a cell's color
    """

    def __init__(self, unique_id, model, pos, initial_color: int):
        """
        Create a cell, in the given state, at the given row, col position.
        """
        super().__init__(unique_id, model)
        # The "pos" variable in mesa is special, so I avoid it here
        self._row = pos[0]
        self._col = pos[1]
        self._color = initial_color  # The cell's current color (int)
        self._next_color = None
        self.agents = []
        self.areas = []
        self.is_border_cell = False

    def __str__(self):
        return (f"Cell ({self.unique_id}, pos={self.position}, "
                f"color={self.color}, num_agents={self.num_agents_in_cell})")

    @property
    def col(self):
        """The col location of this cell."""
        return self._col

    @property
    def row(self):
        """The row location of this cell."""
        return self._row

    @property
    def position(self):  # The variable pos is special in mesa!
        """The location of this cell."""
        return self._row, self._col

    @property
    def color(self):
        """The current color of this cell."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

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

    def advance(self):
        """
        Set the state of the agent to the next state.
        TODO
        """
        # self._color = self._next_color
        pass
