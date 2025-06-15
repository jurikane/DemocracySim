from typing import TYPE_CHECKING, cast, List, Optional
import mesa
from democracy_sim.participation_agent import VoteAgent, ColorCell
from democracy_sim.social_welfare_functions import majority_rule, approval_voting
from democracy_sim.distance_functions import spearman, kendall_tau
from itertools import permutations, product, combinations
from math import sqrt
import numpy as np

# Voting rules to be accessible by index
social_welfare_functions = [majority_rule, approval_voting]
# Distance functions
distance_functions = [spearman, kendall_tau]


class Area(mesa.Agent):
    def __init__(self, unique_id, model, height, width, size_variance):
        """
        Create a new area.

        Attributes:
            unique_id (int): The unique identifier of the area.
            model (ParticipationModel): The simulation model of which the area is part of.
            height (int): The average height of the area (see size_variance).
            width (int): The average width of the area (see size_variance).
            size_variance (float): A variance factor applied to height and width.
        """
        if TYPE_CHECKING:  # Type hint for IDEs
            model = cast(ParticipationModel, model)
        super().__init__(unique_id=unique_id,  model=model)
        self._set_dimensions(width, height, size_variance)
        self.agents = []
        self._personality_distribution = None
        self.cells = []
        self._idx_field = None  # An indexing position of the area in the grid
        self._color_distribution = np.zeros(model.num_colors) # Initialize to 0
        self._voted_ordering = None
        self._voter_turnout = 0  # In percent
        self._dist_to_reality = None  # Elected vs. actual color distribution

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
                cell = self.model.grid.get_cell_list_contents([(x, y)])[0]
                if TYPE_CHECKING:
                    cell = cast(ColorCell, cell)
                self.add_cell(cell)  # Add the cell to the area
                # Add all voting agents to the area
                for agent in cell.agents:
                    self.add_agent(agent)
                cell.add_area(self)  # Add the area to the color-cell
                # Mark as a border cell if true
                if (x_area == 0 or y_area == 0
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
        self._personality_distribution = [p_counts[str(p)] / self.num_agents
                                          for p in personalities]

    def add_agent(self, agent: VoteAgent) -> None:
        """
        Appends an agent to the areas agents list.

        Args:
            agent (VoteAgent): The agent to be added to the area.
        """
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
        if preference_profile.ndim != 2:
            print("Area", self.unique_id, "no one participated in the election")
            return 0  # TODO: What to do in this case? Cease the simulation?
        # Aggregate the preferences ⇒ returns an option ordering
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
        Gathers votes from agents who choose to (and can afford to) participate.

        Each participating agent contributes a vector of dissatisfaction values with
        respect to the available options. These values are combined into a NumPy array.

        Returns:
            np.ndarray: A 2D array representing the preference profiles of all
                participating agents. Each row corresponds to an agent's vote.
        """
        preference_profile = []
        for agent in self.agents:
            model = self.model
            el_costs = model.election_costs
            # Give agents their (new) known fields
            agent.update_known_cells(area=self)
            if (agent.assets >= el_costs
                    and agent.ask_for_participation(area=self)):
                agent.num_elections_participated += 1
                # Collect the participation fee
                agent.assets = agent.assets - el_costs
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
        """
        model = self.model
        # Calculate the distance to the real distribution using distance_func
        real_color_ord = np.argsort(self.color_distribution)[::-1]  # Descending
        dist_func = model.distance_func
        self._dist_to_reality = dist_func(real_color_ord, self.voted_ordering,
                                          model.color_search_pairs)
        # Calculate the rpa - rewards per agent (can be negative)
        rpa = (0.5 - self.dist_to_reality) * model.max_reward  # TODO: change this (?)
        # Distribute the two types of rewards
        color_search_pairs = model.color_search_pairs
        for a in self.agents:
            # Personality-based reward factor
            p = dist_func(a.personality, real_color_ord, color_search_pairs)
            # + common reward (reward_pa) for all agents
            a.assets = int(a.assets + (0.5 - p) * model.max_reward + rpa)
            if a.assets < 0:  # Correct wealth if it fell below zero
                a.assets = 0

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
        colors = np.random.choice(self.voted_ordering, size=n_to_mutate,
                                  p=self.model.color_probs)
        # Assign the newly selected colors to the cells
        for cell, color in zip(cells_to_mutate, colors):
            cell.color = color
        # Important: Update the color distribution (because colors changed)
        self._update_color_distribution()


def compute_collective_assets(model):
    sum_assets = sum(agent.assets for agent in model.voting_agents)
    return sum_assets


def compute_gini_index(model):
    # TODO: separate to be able to calculate it zone-wise as well as globally
    # TODO: Unit-test this function
    # Extract the list of assets for all agents
    assets = [agent.assets for agent in model.voting_agents]
    n = len(assets)
    if n == 0:
        return 0  # No agents, no inequality
    # Sort the assets
    sorted_assets = sorted(assets)
    # Calculate the Gini Index
    cumulative_sum = sum((i + 1) * sorted_assets[i] for i in range(n))
    total_sum = sum(sorted_assets)
    if total_sum == 0:
        return 0  # No agent has any assets => view as total equality
    gini_index = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
    return int(gini_index * 100)  # Return in "percent" (0-100)


def get_voter_turnout(model):
    voter_turnout_sum = 0
    num_areas = model.num_areas
    for area in model.areas:
        voter_turnout_sum += area.voter_turnout
    if not model.global_area is None:
        # TODO: Check the correctness and whether it makes sense to include the global area here
        voter_turnout_sum += model.global_area.voter_turnout
        num_areas += 1
    elif num_areas == 0:
        return 0
    return voter_turnout_sum / num_areas


def create_personality(num_colors):
    """ NOT USED
    Creates and returns a list of 'personalities' that are to be assigned
    to agents. Each personality is a NumPy array of length 'num_colors'
    but it is not a full ranking vector since the number of colors influencing
    the personality is limited. The array is therefore not normalized.
    White (color 0) is never part of a personality.

    Args:
        num_colors: The number of colors in the simulation.
    """
    # TODO add unit tests for this function
    personality = np.random.randint(0, 100, num_colors)  # TODO low=0 or 1?
    # Save the sum to "normalize" the values later (no real normalization)
    sum_value = sum(personality) + 1e-8  # To avoid division by zero
    # Select only as many features as needed (num_personality_colors)
    # to_del = num_colors - num_personality_colors  # How many to be deleted
    # if to_del > 0:
    #     # The 'replace=False' ensures that indexes aren't chosen twice
    #     indices = np.random.choice(num_colors, to_del, replace=False)
    #     personality[indices] = 0  # 'Delete' the values
    personality[0] = 0  # White is never part of the personality
    # "Normalize" the rest of the values
    personality = personality / sum_value
    return personality


def get_color_distribution_function(color):
    """
    This method returns a lambda function for the color distribution chart.

    Args:
        color: The color number (used as index).
    """
    return lambda m: m.av_area_color_dst[color]


def get_area_voter_turnout(area):
    if isinstance(area, Area):
        return area.voter_turnout
    return None

def get_area_dist_to_reality(area):
    if isinstance(area, Area):
        return area.dist_to_reality
    return None

def get_area_color_distribution(area):
    if isinstance(area, Area):
        return area.color_distribution.tolist()
    return None

def get_election_results(area):
    """
    Returns the voted ordering as a list or None if not available.

    Returns: 
        List of voted ordering or None.
    """
    if isinstance(area, Area) and area.voted_ordering is not None:
        return area.voted_ordering.tolist()
    return None


class CustomScheduler(mesa.time.BaseScheduler):
    def step(self):
        """
        Execute the step function for all area- and cell-agents by type,
        first for Areas then for ColorCells.
        """
        model = self.model
        if TYPE_CHECKING:
            model = cast(ParticipationModel, model)
        # Step through Area agents first (and in "random" order)
        # TODO think about randomization process
        model.random.shuffle(model.areas)
        for area in model.areas:
            area.step()
        # TODO: add global election?
        self.steps += 1
        self.time += 1


class ParticipationModel(mesa.Model):
    """
    The ParticipationModel class provides a base environment for
    multi-agent simulations within a grid-based world (split into territories)
    that reacts dynamically to frequently held collective decision-making
    processes ("elections"). It incorporates voting agents with personalities,
    color cells (grid fields), and areas (election territories). This model is
    designed to analyze different voting rules and their impact.

    This class provides mechanisms for creating and managing cells, agents,
    and areas, along with data collection for analysis. Colors in the model
    mutate depending on a predefined mutation rate and are influenced by
    elections. Agents interact based on their personalities, knowledge, and
    experiences.

    Attributes:
        grid (mesa.space.SingleGrid): Grid representing the environment
            with a single occupancy per cell (the color).
        height (int): The height of the grid.
        width (int): The width of the grid.
        colors (ndarray): Array containing the unique color identifiers.
        voting_rule (Callable): A function defining the social welfare
            function to aggregate agent preferences. This callable typically
            takes agent rankings as input and returns a single aggregate result.
        distance_func (Callable): A function used to calculate a
            distance metric when comparing rankings. It takes two rankings
            and returns a numeric distance score.
        mu (float): Mutation rate; the probability of each color cell to mutate
            after an elections.
        color_probs (ndarray):
            Probabilities used to determine individual color mutation outcomes.
        options (ndarray): Matrix (array of arrays) where each subarray
            represents an option (color-ranking) available to agents.
        option_vec (ndarray): Array holding the indices of the available options
            for computational efficiency.
        color_cells (list[ColorCell]): List of all color cells.
            Initialized during the model setup.
        voting_agents (list[VoteAgent]): List of all voting agents.
            Initialized during the model setup.
        personalities (list): List of unique personalities available for agents.
        personality_distribution (ndarray): The (global) probability
            distribution of personalities among all agents.
        areas (list[Area]): List of areas (regions or territories within the
            grid) in which elections take place. Initialized during model setup.
        global_area (Area): The area encompassing the entire grid.
        av_area_height (int): Average height of areas in the simulation.
        av_area_width (int): Average width of areas created in the simulation.
        area_size_variance (float): Variance in area sizes to introduce
            non-uniformity among election territories.
        common_assets (int): Total resources to be distributed among all agents.
        av_area_color_dst (ndarray): Current (area)-average color distribution.
        election_costs (float): Cost associated with participating in elections.
        max_reward (float): Maximum reward possible for an agent each election.
        known_cells (int): Number of cells each agent knows the color of.
        datacollector (mesa.DataCollector): A tool for collecting data
            (metrics and statistics) at each simulation step.
        scheduler (CustomScheduler): The scheduler responsible for executing the
            step function.
        draw_borders (bool): Only for visualization (no effect on simulation).
        _preset_color_dst (ndarray): A predefined global color distribution
            (set randomly) that affects cell initialization globally.
        """

    def __init__(self, height, width, num_agents, num_colors, num_personalities,
                 mu, election_impact_on_mutation, common_assets, known_cells,
                 num_areas, av_area_height, av_area_width, area_size_variance,
                 patch_power, color_patches_steps, draw_borders, heterogeneity,
                 rule_idx, distance_idx, election_costs, max_reward,
                 show_area_stats):
        super().__init__()
        # TODO clean up class (public/private variables)
        self.height = height
        self.width = width
        self.colors = np.arange(num_colors)
        # Create a scheduler that goes through areas first then color cells
        self.scheduler = CustomScheduler(self)
        # The grid
        # SingleGrid enforces at most one agent per cell;
        # MultiGrid allows multiple agents to be in the same cell.
        self.grid = mesa.space.SingleGrid(height=height, width=width, torus=True)
        # Random bias factors that affect the initial color distribution
        self._vertical_bias = self.random.uniform(0, 1)
        self._horizontal_bias = self.random.uniform(0, 1)
        self.draw_borders = draw_borders
        # Color distribution (global)
        self._preset_color_dst = self.create_color_distribution(heterogeneity)
        self._av_area_color_dst = self._preset_color_dst
        # Elections
        self.election_costs = election_costs
        self.max_reward = max_reward
        self.known_cells = known_cells  # Integer
        self.voting_rule = social_welfare_functions[rule_idx]
        self.distance_func = distance_functions[distance_idx]
        self.options = self.create_all_options(num_colors)
        # Simulation variables
        self.mu = mu  # Mutation rate for the color cells (0.1 = 10 % mutate)
        self.common_assets = common_assets
        # Election impact factor on color mutation through a probability array
        self.color_probs = self.init_color_probs(election_impact_on_mutation)
        # Create search pairs once for faster iterations when comparing rankings
        self.search_pairs = list(combinations(range(0, self.options.size), 2))  # TODO check if correct!
        self.option_vec = np.arange(self.options.size)  # Also to speed up
        self.color_search_pairs = list(combinations(range(0, num_colors), 2))
        # Create color cells
        self.color_cells: List[Optional[ColorCell]] = [None] * (height * width)
        self._initialize_color_cells()
        # Create agents
        # TODO: Where do the agents get there known cells from and how!?
        self.voting_agents: List[Optional[VoteAgent]] = [None] * num_agents
        self.personalities = self.create_personalities(num_personalities)
        self.personality_distribution = self.pers_dist(num_personalities)
        self.initialize_voting_agents()
        # Area variables
        self.global_area = self.initialize_global_area()  # TODO create bool variable to make this optional
        self.areas: List[Optional[Area]] = [None] * num_areas
        self.av_area_height = av_area_height
        self.av_area_width = av_area_width
        self.area_size_variance = area_size_variance
        # Adjust the color pattern to make it less random (see color patches)
        self.adjust_color_pattern(color_patches_steps, patch_power)
        # Create areas
        self.initialize_all_areas()
        # Data collector
        self.datacollector = self.initialize_datacollector()
        # Collect initial data
        self.datacollector.collect(self)
        # Statistics
        self.show_area_stats = show_area_stats

    @property
    def num_colors(self):
        return len(self.colors)

    @property
    def av_area_color_dst(self):
        return self._av_area_color_dst

    @av_area_color_dst.setter
    def av_area_color_dst(self, value):
        self._av_area_color_dst = value

    @property
    def num_agents(self):
        return len(self.voting_agents)

    @property
    def num_areas(self):
        return len(self.areas)

    @property
    def preset_color_dst(self):
        return len(self._preset_color_dst)

    def _initialize_color_cells(self):
        """
        This method initializes a color cells for each cell in the model's grid.
        """
        # Create a color cell for each cell in the grid
        for unique_id, (_, (row, col)) in enumerate(self.grid.coord_iter()):
            # The colors are chosen by a predefined color distribution
            color = self.color_by_dst(self._preset_color_dst)
            # Create the cell
            cell = ColorCell(unique_id, self, (row, col), color)
            # Add it to the grid
            self.grid.place_agent(cell, (row, col))
            # Add the color cell to the scheduler
            #self.scheduler.add(cell) # TODO: check speed diffs using this..
            # And to the 'model.color_cells' list (for faster access)
            self.color_cells[unique_id] = cell  # TODO: check if its not better to simply use the grid when finally changing the grid type to SingleGrid

    def initialize_voting_agents(self):
        """
        This method initializes as many voting agents as set in the model with
        a randomly chosen personality. It places them randomly on the grid.
        It also ensures that each agent is assigned to the color cell it is
        standing on.
        """
        dist = self.personality_distribution
        rng = np.random.default_rng()
        assets = self.common_assets // self.num_agents
        for a_id in range(self.num_agents):
            # Get a random position
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            personality = rng.choice(self.personalities, p=dist)
            # Create agent without appending (add to the pre-defined list)
            agent = VoteAgent(a_id, self, (x, y), personality,
                              assets=assets, add=False)  # TODO: initial assets?!
            self.voting_agents[a_id] = agent  # Add using the index (faster)
            # Add the agent to the grid by placing it on a cell
            cell = self.grid.get_cell_list_contents([(x, y)])[0]
            if TYPE_CHECKING:
                cell = cast(ColorCell, cell)
            cell.add_agent(agent)

    def init_color_probs(self, election_impact):
        """
        This method initializes a probability array for the mutation of colors.
        The probabilities reflect the election outcome with some impact factor.

        Args:
            election_impact (float): The impact the election has on the mutation.
        """
        p = (np.arange(self.num_colors, 0, -1)) ** election_impact
        # Normalize
        p = p / sum(p)
        return p

    def initialize_area(self, a_id: int, x_coord, y_coord):
        """
        This method initializes one area in the models' grid.
        """
        area = Area(a_id, self, self.av_area_height, self.av_area_width,
                    self.area_size_variance)
        # Place the area in the grid using its indexing field
        # this adds the corresponding color cells and voting agents to the area
        area.idx_field = (x_coord, y_coord)
        # Save in the models' areas-list
        self.areas[a_id] = area

    def initialize_all_areas(self) -> None:
        """
        Initializes all areas on the grid in the model.

        This method divides the grid into approximately evenly distributed areas,
        ensuring that the areas are spaced as uniformly as possible based
        on the grid dimensions and the average area size specified by
        `av_area_width` and `av_area_height`.

        The grid may contain more or fewer areas than an exact square
        grid arrangement due to `num_areas` not always being a perfect square.
        If the number of areas is not a perfect square, the remaining areas
        are placed randomly on the grid to ensure that `num_areas`
        areas are initialized.

        Args:
            None.

        Returns:
            None. initializes `num_areas` and places them directly on the grid.

        Raises:
            None, but if `self.num_areas == 0`, the method exits early.

        Example:
            - Given `num_areas = 4` and `grid.width = grid.height = 10`,
              this method might initialize areas with approximate distances
              to maximize uniform distribution (like a 2x2 grid).
            - For `num_areas = 5`, four areas will be initialized evenly, and
              the fifth will be placed randomly due to the uneven distribution.
        """
        if self.num_areas == 0:
            return
        # Calculate the number of areas in each direction
        roo_apx = round(sqrt(self.num_areas))
        nr_areas_x = self.grid.width // self.av_area_width
        nr_areas_y = self.grid.width // self.av_area_height
        # Calculate the distance between the areas
        area_x_dist = self.grid.width // roo_apx
        area_y_dist = self.grid.height // roo_apx
        print(f"roo_apx: {roo_apx}, nr_areas_x: {nr_areas_x}, "
              f"nr_areas_y: {nr_areas_y}, area_x_dist: {area_x_dist}, "
              f"area_y_dist: {area_y_dist}")  # TODO rm print
        x_coords = range(0, self.grid.width, area_x_dist)
        y_coords = range(0, self.grid.height, area_y_dist)
        # Add additional areas if necessary (num_areas not a square number)
        additional_x, additional_y = [], []
        missing = self.num_areas - len(x_coords) * len(y_coords)
        for _ in range(missing):
            additional_x.append(self.random.randrange(self.grid.width))
            additional_y.append(self.random.randrange(self.grid.height))
        # Create the area's ids
        a_ids = iter(range(self.num_areas))
        # Initialize all areas
        for x_coord in x_coords:
            for y_coord in y_coords:
                a_id = next(a_ids, -1)
                if a_id == -1:
                    break
                self.initialize_area(a_id, x_coord, y_coord)
        for x_coord, y_coord in zip(additional_x, additional_y):
            self.initialize_area(next(a_ids), x_coord, y_coord)


    def initialize_global_area(self):
        """
        This method initializes the global area spanning the whole grid.

        Returns:
            Area: The global area (with unique_id set to -1 and idx to (0, 0)).
        """
        global_area = Area(-1, self, self.height, self.width, 0)
        # Place the area in the grid using its indexing field
        # this adds the corresponding color cells and voting agents to the area
        global_area.idx_field = (0, 0)
        return global_area


    def create_personalities(self, n: int):
        """
        Creates n unique "personalities," where a "personality" is a specific
        permutation of self.num_colors color indices.

        Args:
            n (int): Number of unique personalities to generate.

        Returns:
            np.ndarray: Array of shape `(n, num_colors)`.

        Raises:
            ValueError: If `n` exceeds the possible unique permutations.

        Example:
            for n=2 and self.num_colors=3, the function could return:

            [[1, 0, 2],
            [2, 1, 0]]
        """
        # p_colors = range(1, self.num_colors)  # Personalities exclude white
        max_permutations = np.math.factorial(self.num_colors)
        if n > max_permutations or n < 1:
            raise ValueError(f"Cannot generate {n} unique personalities: "
                             f"only {max_permutations} unique ones exist.")
        selected_permutations = set()
        while len(selected_permutations) < n:
            # Sample a permutation lazily and add it to the set
            perm = tuple(self.random.sample(range(self.num_colors),
                                            self.num_colors))
            selected_permutations.add(perm)

        return np.array(list(selected_permutations))


    def initialize_datacollector(self):
        color_data = {f"Color {i}": get_color_distribution_function(i) for i in
                      range(self.num_colors)}
        return mesa.DataCollector(
            model_reporters={
                "Collective assets": compute_collective_assets,
                "Gini Index (0-100)": compute_gini_index,
                "Voter turnout globally (in percent)": get_voter_turnout,
                **color_data
            },
            agent_reporters={
                # "Voter Turnout": lambda a: a.voter_turnout if isinstance(a, Area) else None,
                # "Color Distribution": lambda a: a.color_distribution if isinstance(a, Area) else None,
                #
                #"VoterTurnout": lambda a: a.voter_turnout if isinstance(a, Area) else None,
                "VoterTurnout": get_area_voter_turnout,
                "DistToReality": get_area_dist_to_reality,
                "ColorDistribution": get_area_color_distribution,
                "ElectionResults": get_election_results,
                # "Personality-Based Reward": get_area_personality_based_reward,
                # "Gini Index": get_area_gini_index
            },
            # tables={
            #    "AreaData": ["Step", "AreaID", "ColorDistribution",
            #                 "VoterTurnout"]
            # }
        )


    def step(self):
        """
        Advance the model by one step.
        """

        # Conduct elections in the areas
        # and then mutate the color cells according to election outcomes
        self.scheduler.step()
        # Update the global color distribution
        self.update_av_area_color_dst()
        # Collect data for monitoring and data analysis
        self.datacollector.collect(self)


    def adjust_color_pattern(self, color_patches_steps: int, patch_power: float):
        """Adjusting the color pattern to make it less predictable.

        Args:
            color_patches_steps: How often to run the color-patches step.
            patch_power: The power of the patching (like a radius of impact).
        """
        cells = self.color_cells
        for _ in range(color_patches_steps):
            print(f"Color adjustment step {_}")
            self.random.shuffle(cells)
            for cell in cells:
                most_common_color = self.color_patches(cell, patch_power)
                cell.color = most_common_color


    def create_color_distribution(self, heterogeneity: float):
        """
        This method is used to create a color distribution that has a bias
        according to the given heterogeneity factor.

        Args:
            heterogeneity (float): Factor used as sigma in 'random.gauss'.
        """
        colors = range(self.num_colors)
        values = [abs(self.random.gauss(1, heterogeneity)) for _ in colors]
        # Normalize (with float division)
        total = sum(values)
        dst_array = [value / total for value in values]
        return dst_array


    def color_patches(self, cell: ColorCell, patch_power: float):
        """
        This method is used to create a less random initial color distribution
        using a similar logic to the color patches model.
        It uses a (normalized) bias coordinate to center the impact of the
        color patches structures impact around.

        Args:
            cell: The cell that may change its color accordingly
            patch_power: Like a radius of impact around the bias point.

        Returns:
            int: The consensus color or the cell's own color if no consensus.
        """
        # Calculate the normalized position of the cell
        normalized_x = cell.row / self.height
        normalized_y = cell.col / self.width
        # Calculate the distance of the cell to the bias point
        bias_factor = (abs(normalized_x - self._horizontal_bias)
                       + abs(normalized_y - self._vertical_bias))
        # The closer the cell to the bias-point, the less often it is
        # to be replaced by a color chosen from the initial distribution:
        if abs(self.random.gauss(0, patch_power)) < bias_factor:
            return self.color_by_dst(self._preset_color_dst)
        # Otherwise, apply the color patches logic
        neighbor_cells = self.grid.get_neighbors((cell.row, cell.col),
                                                 moore=True,
                                                 include_center=False)
        color_counts = {}  # Count neighbors' colors
        for neighbor in neighbor_cells:
            if isinstance(neighbor, ColorCell):
                color = neighbor.color
                color_counts[color] = color_counts.get(color, 0) + 1
        if color_counts:
            max_count = max(color_counts.values())
            most_common_colors = [color for color, count in color_counts.items()
                                  if count == max_count]
            return self.random.choice(most_common_colors)
        return cell.color  # Return the cell's own color if no consensus


    def update_av_area_color_dst(self):
        """
        This method updates the av_area_color_dst attribute of the model.
        Beware: On overlapping areas, cells are counted several times.
        """
        sums = np.zeros(self.num_colors)
        for area in self.areas:
            sums += area.color_distribution
        # Return the average color distributions
        self.av_area_color_dst = sums / self.num_areas


    @staticmethod
    def pers_dist(size):
        """
        This method creates a normalized normal distribution array for picking
        and depicting the distribution of personalities in the model.

        Args:
            size: The mean value of the normal distribution.

        Returns:
            np.array: Normalized (sum is one) array mimicking a gaussian curve.
        """
        # Generate a normal distribution
        rng = np.random.default_rng()
        dist = rng.normal(0, 1, size)
        dist.sort()  # To create a gaussian curve like array
        dist = np.abs(dist)  # Flip negative values "up"
        # Normalize the distribution to sum to one
        dist /= dist.sum()
        return dist


    @staticmethod
    def create_all_options(n: int, include_ties=False):
        """
        Creates a matrix (an array of all possible ranking vectors),
        if specified including ties.
        Rank values start from 0.

        Args:
            n (int): The number of items to rank (number of colors in our case)
            include_ties (bool): If True, rankings include ties.

        Returns:
            np.array: A matrix containing all possible ranking vectors.
        """
        if include_ties:
            # Create all possible combinations and sort out invalid rankings
            # i.e. [1, 1, 1] or [1, 2, 2] aren't valid as no option is ranked first.
            r = np.array([np.array(comb) for comb in product(range(n), repeat=n)
                          if set(range(max(comb))).issubset(comb)])
        else:
            r = np.array([np.array(p) for p in permutations(range(n))])
        return r

    @staticmethod
    def color_by_dst(color_distribution: np.array) -> int:
        """
        Selects a color (int) from range(len(color_distribution))
        based on the given color_distribution array, where each entry represents
        the probability of selecting that index.

        Args:
            color_distribution: Array determining the selection probabilities.

        Returns:
            int: The selected index based on the given probabilities.

        Example:
            color_distribution = [0.2, 0.3, 0.5]
            Color 1 will be selected with a probability of 0.3.
        """
        if abs(sum(color_distribution) -1) > 1e-8:
            raise ValueError("The color_distribution array must sum to 1.")
        r = np.random.random()  # Random float between 0 and 1
        cumulative_sum = 0.0
        for color_idx, prob in enumerate(color_distribution):
            if prob < 0:
                raise ValueError("color_distribution contains negative value.")
            cumulative_sum += prob
            if r < cumulative_sum:  # Compare r against the cumulative probability
                return color_idx

        # This point should never be reached.
        raise ValueError("Unexpected error in color_distribution.")
