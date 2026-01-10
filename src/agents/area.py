from __future__ import annotations
from typing import TYPE_CHECKING, cast, List
import numpy as np
from mesa import Agent
if TYPE_CHECKING:  # Type hint for IDEs
    from src.models.participation_model import ParticipationModel
    from src.agents.color_cell import ColorCell
    from src.agents.vote_agent import VoteAgent


class Area(Agent):
    """
    While technically an agent, this class contains major parts of the simulation logic.
    An area containing agents and cells, and is conducting the elections.
    """
    def __init__(self, unique_id, model: ParticipationModel,
                 height, width, size_variance):
        """
        Create a new area.

        Attributes:
            unique_id (int): The unique identifier of the area.
            model (ParticipationModel): The simulation model of which the area is part of.
            height (int): The average height of the area (see size_variance).
            width (int): The average width of the area (see size_variance).
            size_variance (float): A variance factor applied to height and width.
        """
        super().__init__(unique_id=unique_id,  model=model)
        self._set_dimensions(width, height, size_variance)
        self.agents: List["VoteAgent"] = []
        self._personality_distribution = None
        self.cells: List["ColorCell"] = []
        self._idx_field = None  # An indexing position of the area in the grid
        self._color_distribution = np.zeros(model.num_colors) # Initialize to 0
        self._voted_ordering = None
        self._voter_turnout = 0  # In percent
        self._dist_to_reality = None  # Elected vs. actual color distribution
        self._election_fee_pool: int = 0

    def __str__(self):
        return (f"Area(id={self.unique_id}, size={self._height}x{self._width}, "
                f"at idx_field={self._idx_field}, "
                f"num_agents={self.num_agents}, num_cells={self.num_cells}, "
                f"color_distribution={self.color_distribution})")

    def _set_dimensions(self, width, height, size_var):
        """
        Sets the area's dimensions based on the provided width, height, and variance factor.

        This function adjusts the width and height by a random factor drawn from
        the range [1 - size_var, 1 + size_var]. If size_var is zero, no variance
        is applied.

        Args:
            width (int): The average width of the area.
            height (int): The average height of the area.
            size_var (float): A variance factor applied to width and height.
                Must be in [0, 1].

        Raises:
            ValueError: If size_var is not between 0 and 1.
        """
        if size_var == 0:
            self._width = width
            self._height = height
            self.width_off, self.height_off = 0, 0
        elif size_var > 1 or size_var < 0:
            raise ValueError("Size variance must be between 0 and 1")
        else:  # Apply variance
            w_var_factor = self.random.uniform(1 - size_var, 1 + size_var)
            h_var_factor = self.random.uniform(1 - size_var, 1 + size_var)
            self._width = int(width * w_var_factor)
            self.width_off = abs(width - self._width)
            self._height = int(height * h_var_factor)
            self.height_off = abs(height - self._height)

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def num_cells(self):
        return self._width * self._height

    @property
    def personality_distribution(self):
        return self._personality_distribution

    @property
    def color_distribution(self):
        return self._color_distribution

    @property
    def voted_ordering(self):
        return self._voted_ordering

    @property
    def voter_turnout(self):
        return self._voter_turnout

    @property
    def dist_to_reality(self):
        return self._dist_to_reality

    @property
    def idx_field(self):
        return self._idx_field

    @idx_field.setter
    def idx_field(self, pos: tuple):
        """
        Sets the indexing field (cell coordinate in the grid) of the area.

        This method sets the areas indexing-field (top-left cell coordinate)
        which determines which cells and agents on the grid belong to the area.
        The cells and agents are added to the area's lists of cells and agents.

        Args:
            pos: (x, y) representing the areas top-left coordinates.
        """
        # TODO: Check - isn't it better to make sure agents are added to the area when they are created?
        # TODO -- There is something wrong here!!! (Agents are not added to the areas)
        if TYPE_CHECKING:  # Type hint for IDEs
            self.model = cast(ParticipationModel, self.model)
        try:
            x_val, y_val = pos
        except ValueError:
            raise ValueError("The idx_field must be a tuple")
        # Check if the values are within the grid
        if x_val < 0 or x_val >= self.model.width:
            raise ValueError(f"The x={x_val} value must be within the grid")
        if y_val < 0 or y_val >= self.model.height:
            raise ValueError(f"The y={y_val} value must be within the grid")
        x_off = self.width_off // 2
        y_off = self.height_off // 2
        # Adjusting indices with offset and ensuring they wrap around the grid
        adjusted_x = (x_val + x_off) % self.model.width
        adjusted_y = (y_val + y_off) % self.model.height
        # Assign the cells to the area
        for x_area in range(self._width):
            for y_area in range(self._height):
                x = (adjusted_x + x_area) % self.model.width
                y = (adjusted_y + y_area) % self.model.height
                contents = self.model.grid.get_cell_list_contents([(x, y)])
                if not contents:
                    raise RuntimeError(
                        f"Grid cell ({x},{y}) is empty – expected a ColorCell.")
                cell = contents[0]
                if TYPE_CHECKING:
                    cell = cast(ColorCell, cell)
                self.add_cell(cell)  # Add the cell to the area
                # Add all voting agents to the area
                for agent in cell.agents:
                    self.add_agent(agent)
                cell.add_area(self)  # Add the area to the color-cell
                # Mark as a border cell if true, but not for the global area
                if self.unique_id != -1 and (x_area == 0 or y_area == 0
                        or x_area == self._width - 1
                        or y_area == self._height - 1):
                    cell.is_border_cell = True
        self._idx_field = (adjusted_x, adjusted_y)
        self._update_color_distribution()
        self._update_personality_distribution()

    def _update_personality_distribution(self) -> None:
        """
        This method calculates the areas current distribution of personalities.
        """
        personalities = list(self.model.personalities)
        p_counts = {str(i): 0 for i in personalities}
        # Count the occurrence of each personality
        for agent in self.agents:
            p_counts[str(agent.personality)] += 1
        # Normalize the counts
        if self.num_agents == 0:
            self._personality_distribution = [0 for _ in personalities]
        else:
            self._personality_distribution = [p_counts[str(p)] / self.num_agents
                                              for p in personalities]

    def add_agent(self, agent: VoteAgent) -> None:
        """
        Appends an agent to the areas agents list.

        Args:
            agent (VoteAgent): The agent to be added to the area.
        """
        # Make sure its an instance of Agent
        if not isinstance(agent, Agent):
            raise ValueError("Only VoteAgent instances can be added to an Area")
        self.agents.append(agent)

    def add_cell(self, cell: ColorCell) -> None:
        """
        Appends a cell to the areas cells list.

        Args:
            cell (ColorCell): The agent to be added to the area.
        """
        self.cells.append(cell)


    def _conduct_election(self) -> int:
        """
        Simulates the election within the area and manages rewards.

        The election process asks agents to participate, collects votes,
        aggregates preferences using the model's voting rule,
        and saves the elected option as the latest winning option.
        Agents incur costs for participation
        and may receive rewards based on the outcome.

        Returns:
            int: The voter turnout in percent. Returns 0 if no agent participates.
        """
        # Ask agents for participation and their votes
        preference_profile = self._tally_votes()
        # Check for the case that no agent participated
        if preference_profile.ndim != 2 or preference_profile.shape[0] == 0:
            # Set to previous outcome but don't distribute rewards as usual
            print("Area", self.unique_id, "no one participated in the election")
            # If no previous outcome, use the real distribution ordering
            real_color_ord = np.argsort(self.color_distribution)[::-1]
            if self._voted_ordering is None:
                self._voted_ordering = real_color_ord
            # Update dist_to_reality for monitoring but no rewards
            self._dist_to_reality = self.model.distance_func(
                real_color_ord, self._voted_ordering,
                self.model.color_search_pairs
            )
            # Slightly punish for non-participation
            for a in self.agents:
                a.reward_agent(-1)
            return 0
        # Aggregate the preferences ⇒ returns an option ordering (indices into options)
        aggregated = self.model.voting_rule(preference_profile)
        # Save the "elected" ordering in self._voted_ordering
        winning_option = aggregated[0]
        self._voted_ordering = self.model.options[winning_option]
        # Calculate and distribute rewards
        self._distribute_rewards()
        # TODO check whether the current color dist and the mutation of the
        #  colors is calculated and applied correctly and does not interfere
        #  in any way with the election process
        # Statistics
        n = preference_profile.shape[0]  # Number agents participated
        return int((n / self.num_agents) * 100) # Voter turnout in percent

    def _tally_votes(self):
        """
        Gathers votes from agents who choose to participate.

        Each participating agent contributes a vector of dissatisfaction values with
        respect to the available options. These values are combined into a NumPy array.

        Returns:
            np.ndarray: A 2D array representing the preference profiles of all
                participating agents. Each row corresponds to an agent's vote.
        """
        preference_profile = []
        # Reset pool for this election step.
        self._election_fee_pool = 0
        el_cost_rate = self.model.election_costs
        for agent in self.agents:
            # election_costs is treated as a percent (0..100) of current assets.
            cost = int(agent.assets * el_cost_rate)
            # Ensure participating agents pay at least 1 if they have assets.
            if cost == 0 and agent.assets >= 1 and not el_cost_rate == 0:
                cost = 1
            # Give agents their (new) known fields
            agent.update_known_cells(area=self)
            if agent.ask_for_participation(area=self):
                agent.num_elections_participated += 1
                # Collect the participation fee into the area pool
                agent.assets = agent.assets - cost
                self._election_fee_pool += cost
                # Ask the agent for her preference
                preference_profile.append(agent.vote(area=self))
                # agent.vote returns an array containing dissatisfaction values
                # between 0 and 1 for each option, interpretable as rank values.
        return np.array(preference_profile)

    def _distribute_rewards(self) -> None:
        """
        Calculates and distributes rewards (or penalties) to agents based on outcomes.

        The function measures the difference between the actual color distribution
        and the elected outcome using a distance metric. It then increments or reduces
        agent assets accordingly, ensuring assets do not fall below zero.
        Rewards are budget-balanced around the fees collected from participants
        in the same election step.
        """
        model = self.model
        # Calculate the distance to the real distribution using distance_func
        real_color_ord = np.argsort(self.color_distribution)[::-1]  # Descending
        dist_func = model.distance_func
        self._dist_to_reality = dist_func(
            real_color_ord, self.voted_ordering, model.color_search_pairs
        )
        # Reward budget pool: collected election fees from participating agents
        pool = self._election_fee_pool
        if pool == 0:  # To avoid division by zero.
            pool = 1  # In case election is free (non-participation handled earlier)
        pool_share = pool / self.num_agents  # Equal share per agent
        # Adjust pool to be budget-balanced
        # Distribute the two types of rewards, the common component:
        #   If dist_to_reality large (vote for change), society invests in change.
        #   If it is small (vote for small/no change), society reaps returns.
        common_component = (0.5 - self.dist_to_reality) * pool_share
        color_search_pairs = model.color_search_pairs
        for a in self.agents:
            # Personality-based reward factor
            #   the closer the elected outcome to the agents personality.
            #   the higher the reward for the agent.
            p = dist_func(a.personality, self.voted_ordering, color_search_pairs)
            pers_component = (1 - p) * pool_share
            a.reward_agent(pers_component + common_component)

    def _update_color_distribution(self) -> None:
        """
        Recalculates the area's color distribution and updates the _color_distribution attribute.

        This method counts how many cells of each color belong to the area, normalizes
        the counts by the total number of cells, and stores the result internally.
        """
        color_count = {}
        for cell in self.cells:
            color = cell.color
            color_count[color] = color_count.get(color, 0) + 1
        for color in range(self.model.num_colors):
            dist_val = color_count.get(color, 0) / self.num_cells  # Float
            self._color_distribution[color] = dist_val

    def _filter_cells(self, cell_list):
        """
        This method is used to filter a given list of cells to return only
        those which are within the area.

        Args:
            cell_list: A list of ColorCell cells to be filtered.

        Returns:
            A list of ColorCell cells that are within the area.
        """
        cell_set = set(self.cells)
        return [c for c in cell_list if c in cell_set]

    def step(self) -> None:
        """
        Run one step of the simulation.

        Conduct an election in the area,
        mutate the cells' colors according to the election outcome
        and update the color distribution of the area.
        """
        self._voter_turnout = self._conduct_election()  # The main election logic!
        if self.voter_turnout == 0:
            return  # TODO: What to do if no agent participated..?

        # Mutate colors in cells
        # Take some number of cells to mutate (i.e., 5 %)
        n_to_mutate = int(self.model.mu * self.num_cells)
        # TODO/Idea: What if the voter_turnout determines the mutation rate?
        # randomly select x cells
        cells_to_mutate = self.random.sample(self.cells, n_to_mutate)
        # Use voted ordering to pick colors in descending order
        # To pre-select colors for all cells to mutate
        # TODO: Think about this: should we take local color-structure
        #  into account - like in color patches - to avoid colors mutating into
        #  very random structures? # Middendorf
        colors = self.model.np_random.choice(self.voted_ordering,
                                             size=n_to_mutate,
                                             p=self.model.color_probs)
        # Assign the newly selected colors to the cells
        for cell, color in zip(cells_to_mutate, colors):
            cell.color = color
        # Important: Update the color distribution (because colors changed)
        self._update_color_distribution()
