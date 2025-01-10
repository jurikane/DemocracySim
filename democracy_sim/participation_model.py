import itertools
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
        :param unique_id: The unique identifier of the area.
        :param model: The simulation model of which the area is part of.
        :param height: The average height of the area (see size_variance).
        :param width: The average width of the area (see size_variance).
        :param size_variance: A variance factor applied to height and width.
        """
        if TYPE_CHECKING:  # Type hint for IDEs
            model = cast(ParticipationModel, model)
        super().__init__(unique_id=unique_id,  model=model)
        self._set_dimensions(width, height, size_variance)
        self.agents = []
        self.cells = []
        self._idx_field = None  # An indexing position of the area in the grid
        self.color_distribution = np.zeros(model.num_colors) # Initialize to 0
        self.voted_ordering = None
        self.voter_turnout = 0  # In percent
        self.dist_to_reality = None  # Elected vs. actual color distribution
        self.personality_dist = None

    def __str__(self):
        return (f"Area(id={self.unique_id}, size={self._height}x{self._width}, "
                f"at idx_field={self._idx_field}, "
                f"num_agents={self.num_agents}, num_cells={self.num_cells}, "
                f"color_distribution={self.color_distribution})")

    def _set_dimensions(self, width, height, size_var):
        """
        Set the dimensions of the area right, based on the size variance.
        :param width: The average width of the area.
        :param height: The average height of the area.
        :param size_var: A variance factor applied to height and width.
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
    def idx_field(self):
        return self._idx_field

    @idx_field.setter
    def idx_field(self, pos: tuple):
        """
        This method sets the areas indexing-field (top-left cell coordinate)
        which determines which cells and agents on the grid belong to the area.
        The cells and agents are added to the area's lists of cells and agents.
        :param pos: (x, y) representing the areas top-left coordinates.
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
        self.update_color_distribution()
        self.update_personality_dist()

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_cell(self, cell):
        self.cells.append(cell)


    def conduct_election(self):
        """
        This method holds the primary logic of the simulation by simulating
        the election in the area as well as handling the payments and rewards.
        :return voter_turnout: The percentage of agents that participated.
        """
        model = self.model
        el_costs = model.election_costs
        dist_func = model.distance_func
        # Ask agents for participation
        preference_profile = []
        for agent in self.agents:
            # Give agents their (new) known fields
            agent.update_known_cells(area=self)
            if (agent.assets >= el_costs
                    and agent.ask_for_participation(area=self)):
                agent.num_elections_participated += 1
                # Collect the participation fee from the agents
                agent.assets = agent.assets - el_costs
                # Ask participating agents for their prefs
                preference_profile.append(agent.vote(area=self))
                # agent.vote returns an array containing dissatisfaction values
                # between 0 and 1 for each option, interpretable as rank values.
        preference_profile = np.array(preference_profile)
        # Check for the case that no agent participated
        if preference_profile.ndim != 2:
            print("Area", self.unique_id, "no one participated in the election")
            return 0  # TODO: What to do in this case? Cease the simulation?
        # Aggregate the prefs using the v-rule ⇒ returns an option ordering
        aggregated = model.voting_rule(preference_profile)
        # Save the "elected" ordering in self.voted_ordering
        winning_option = aggregated[0]
        self.voted_ordering = model.options[winning_option]
        # Calculate the distance to the real distribution using distance_func
        real_color_ord = np.argsort(self.color_distribution)[::-1]  # Descending
        self.dist_to_reality = dist_func(real_color_ord, self.voted_ordering,
                                         model.color_search_pairs)
        # Calculate the rpa - rewards per agent (can be negative)
        rpa = (0.5 - self.dist_to_reality) * model.max_reward  # TODO: change this (?)
        # Distribute the two types of rewards
        color_search_pairs = model.color_search_pairs
        for agent in self.agents:
            # Personality-based reward factor
            p = dist_func(agent.personality, real_color_ord, color_search_pairs)
            # + Common reward (reward_pa) for all agents
            agent.assets = int(agent.assets + (0.5-p) * model.max_reward + rpa)
            # Correct wealth if below zero
            if agent.assets < 0:
                agent.assets = 0
        # TODO check whether the current color dist and the mutation of the colors is calculated and applied correctly and does not interfere in any way with the election process
        # Statistics
        n = preference_profile.shape[0]  # Number agents participated
        return int((n / self.num_agents) * 100) # Voter turnout in percent

    def update_color_distribution(self):
        """
        This method calculates the current color distribution of the area
        and saves it in the color_distribution attribute.
        """
        color_count = {}
        for cell in self.cells:
            color = cell.color
            color_count[color] = color_count.get(color, 0) + 1
        for color in range(self.model.num_colors):
            dist_val = color_count.get(color, 0) / self.num_cells  # Float
            self.color_distribution[color] = dist_val

    def update_personality_dist(self):
        """
        This method calculates the areas current distribution of personalities.
        """
        personalities = list(self.model.personalities)
        p_counts = {str(i): 0 for i in personalities}
        # Count the occurrence of each personality
        for agent in self.agents:
            p_counts[str(agent.personality)] += 1
        # Normalize the counts
        self.personality_dist = [p_counts[str(p)] / self.num_agents
                                 for p in personalities]

    def filter_cells(self, cell_list):
        """
        This method is used to filter a given list of cells to return only
        those which are within the area.
        :param cell_list: A list of ColorCell cells to be filtered.
        :return: A list of ColorCell cells that are within the area.
        """
        cell_set = set(self.cells)
        return [c for c in cell_list if c in cell_set]

    def step(self) -> None:
        """
        Conduct an election in the area,
        mutate the cells' colors according to the election outcome
        and update the color distribution of the area.
        """
        self.voter_turnout = self.conduct_election()  # The main election logic!
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
        self.update_color_distribution()


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


def color_by_dst(color_distribution) -> int:
    """
    This method selects a color (int) of range(len(color_distribution))
    such that, each color is selected with a probability according to the
    given color_distribution array.
    Example: color_distribution = [0.2, 0.3, 0.5]
    Color 1 is selected with a probability of 0.3
    """
    r = np.random.random()
    for color_idx, prob in enumerate(color_distribution):
        if r < prob:
            return color_idx
        r -= prob


def create_all_options(n, include_ties=False):
    """
    Creates and returns a matrix (an array of all possible ranking vectors),
    if specified including ties.
    Rank values start from 0.
    :param n: The number of items to rank (number of colors in our case)
    :param include_ties: If True, rankings include ties.
    :return r: A NumPy matrix containing all possible rankings of n items
    """
    if include_ties:
        # Create all possible combinations and sort out invalid rankings
        # i.e. [1, 1, 1] or [1, 2, 2] aren't valid as no option is ranked first.
        r = np.array([np.array(comb) for comb in product(range(n), repeat=n)
                      if set(range(max(comb))).issubset(comb)])
    else:
        r = np.array([np.array(p) for p in permutations(range(n))])
    return r


def create_personality(num_colors):
    """ NOT USED
    Creates and returns a list of 'personalities' that are to be assigned
    to agents. Each personality is a NumPy array of length 'num_colors'
    but it is not a full ranking vector since the number of colors influencing
    the personality is limited. The array is therefore not normalized.
    White (color 0) is never part of a personality.
    :param num_colors: The number of colors in the simulation.
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
    :param color: The color number (used as index).
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

    :return: List of voted ordering or None.
    """
    if isinstance(area, Area) and area.voted_ordering is not None:
        return area.voted_ordering.tolist()
    return None

# def get_area_personality_based_reward(area):
#     # Assuming you have a method to calculate this in the Area class
#     return area.calculate_personality_reward()
#
# def get_area_gini_index(area):
#     # Assuming you have a method to calculate this in the Area class
#     return area.calculate_gini_index()

class CustomScheduler(mesa.time.BaseScheduler):
    def step(self):
        """Execute the step function for all area- and cell-agents by type,
        first for Areas then for ColorCells."""
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
    """A model with some number of agents."""

    def __init__(self, height, width, num_agents, num_colors, num_personalities,
                 mu, election_impact_on_mutation, common_assets, known_cells,
                 num_areas, av_area_height, av_area_width, area_size_variance,
                 patch_power, color_patches_steps, draw_borders, heterogeneity,
                 rule_idx, distance_idx, election_costs, max_reward,
                 show_area_stats):
        super().__init__()
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
        self.vertical_bias = self.random.uniform(0, 1)
        self.horizontal_bias = self.random.uniform(0, 1)
        self.draw_borders = draw_borders
        # Color distribution (global)
        self.color_dst = self.create_color_distribution(heterogeneity)
        self._av_area_color_dst = self.color_dst
        # Elections
        self.election_costs = election_costs
        self.max_reward = max_reward
        self.known_cells = known_cells
        self.voting_rule = social_welfare_functions[rule_idx]
        self.distance_func = distance_functions[distance_idx]
        self.options = create_all_options(num_colors)
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
        self.initialize_color_cells()
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

    def initialize_color_cells(self):
        """
        This method initializes a color cells for each cell in the model's grid.
        """
        # Create a color cell for each cell in the grid
        for unique_id, (_, (row, col)) in enumerate(self.grid.coord_iter()):
            # The colors are chosen by a predefined color distribution
            color = color_by_dst(self.color_dst)
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
        :param election_impact: The impact the election has on the mutation.
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


    def initialize_all_areas(self):
        """
        This method initializes the areas in the models' grid in such a way
        that the areas are spread approximately evenly across the grid.
        Depending on grid size, the number of areas and their (average) sizes.
        TODO create unit tests for this method (Tested manually so far)
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
        """
        global_area = Area(-1, self, self.height, self.width, 0)
        # Place the area in the grid using its indexing field
        # this adds the corresponding color cells and voting agents to the area
        global_area.idx_field = (0, 0)
        return global_area


    def create_personalities(self, n):
        """
        TODO ensure that we end up with n personalities (with unique orderings)
         maybe have to use orderings and convert them
        """
        #p_colors = range(1, self.num_colors)  # Personalities exclude white
        # TODO is it possible to leave out white (--> dist-func)?
        p_colors = list(range(self.num_colors))
        # Make sure the personalities aren't always in the same order
        self.random.shuffle(p_colors)  # Not actually needed but just in case
        personality_options = np.array(list(itertools.permutations(p_colors)))

        if len(personality_options) < n:
            raise ValueError("Not enough unique personality options available.")

        indices = np.random.choice(len(personality_options), n, replace=False)
        selected_personalities = personality_options[indices].copy()

        del personality_options  # Free up memory (variable may be very large)
        return selected_personalities

    def pers_dist(self, size):
        """
        This method creates a normalized normal distribution array for picking
        and depicting the distribution of personalities in the model.
        :param size: The mean value of the normal distribution.
        :return: A normalized (sum is one) array mimicking a gaussian curve.
        """
        # Generate a normal distribution
        rng = np.random.default_rng()
        dist = rng.normal(0, 1, size)
        dist.sort()  # To create a gaussian curve like array
        dist = np.abs(dist)  # Flip negative values "up"
        # Normalize the distribution to sum to one
        dist /= dist.sum()
        return dist


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
        """Advance the model by one step."""

        # Conduct elections in the areas
        # and then mutate the color cells according to election outcomes
        self.scheduler.step()
        # Update the global color distribution
        self.update_av_area_color_dst()
        # Collect data for monitoring and data analysis
        self.datacollector.collect(self)


    def adjust_color_pattern(self, color_patches_steps, patch_power):
        """Adjusting the color pattern to make it less predictable."""
        cells = self.color_cells
        for _ in range(color_patches_steps):
            print(f"Color adjustment step {_}")
            self.random.shuffle(cells)
            for cell in cells:
                most_common_color = self.color_patches(cell, patch_power)
                cell.color = most_common_color

    def create_color_distribution(self, heterogeneity):
        """
        This method is used to create a color distribution that has a bias
        according to the given heterogeneity factor.
        :param heterogeneity: Factor used as sigma in 'random.gauss'.
        """
        colors = range(self.num_colors)
        values = [abs(self.random.gauss(1, heterogeneity)) for _ in colors]
        # Normalize (with float division)
        total = sum(values)
        dst_array = [value / total for value in values]
        return dst_array


    def color_patches(self, cell, patch_power):
        """
        This method is used to create a less random initial color distribution
        using a similar logic to the color patches model.
        It uses a (normalized) bias coordinate to center the impact of the
        color patches structures impact around.
        :param cell: The cell that may change its color accordingly
        :param patch_power: Like a radius of impact around the bias point.
        """
        # Calculate the normalized position of the cell
        normalized_x = cell.row / self.height
        normalized_y = cell.col / self.width
        # Calculate the distance of the cell to the bias point
        bias_factor = (abs(normalized_x - self.horizontal_bias)
                       + abs(normalized_y - self.vertical_bias))
        # The closer the cell to the bias-point, the less often it is
        # to be replaced by a color chosen from the initial distribution:
        if abs(self.random.gauss(0, patch_power)) < bias_factor:
            return color_by_dst(self.color_dst)
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
