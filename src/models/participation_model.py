import random
from typing import TYPE_CHECKING, cast, List, Optional, Callable
import mesa
import numpy as np
from math import factorial
from src.agents import Area, VoteAgent, ColorCell
from src.utils.social_welfare_functions import majority_rule, approval_voting
from src.utils.distance_functions import spearman, kendall_tau
from itertools import permutations, product, combinations
from src.utils.metrics import (compute_gini_index, compute_collective_assets,
                               get_voter_turnout, get_grid_colors,
                               gini_index_0_100)


# Voting rules to be accessible by index
social_welfare_functions = [majority_rule, approval_voting]
# Distance functions
distance_functions = [spearman, kendall_tau]


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
        grid.height (int): The height of the grid.
        grid.width (int): The width of the grid.
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
        known_cells (int): Number of cells each agent knows the color of.
        datacollector (mesa.DataCollector): A tool for collecting data
            (metrics and statistics) at each simulation step.
        scheduler (CustomScheduler): The scheduler responsible for executing the
            step function.
        _preset_color_dst (ndarray): A predefined global color distribution
            (set randomly) that affects cell initialization globally.
    """

    def __init__(self, height, width, num_agents, num_colors, num_personalities,
                 mu, election_impact_on_mutation, common_assets, known_cells,
                 num_areas, av_area_height, av_area_width, area_size_variance,
                 patch_power, color_patches_steps, heterogeneity,
                 rule_idx, distance_idx, election_costs, seed=None,
                 max_steps: Optional[int] = None):
        super().__init__()
        if seed is not None:
            self.random.seed(seed)  # Mesa RNG (Pythons random.Random
            self.np_random = np.random.default_rng(seed)  # Central NumPy RNG
            random.seed(seed)
            np.random.seed(seed)  # For any legacy/global Numpy calls
            print(f"Set models random seed to {seed}")
        else:
            self.np_random = np.random.default_rng()
        # Step control
        self.max_steps: Optional[int] = max_steps
        self.running: bool = True
        # TODO clean up class (public/private variables)
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
        # Color distribution (global)
        self._preset_color_dst = self.create_color_distribution(heterogeneity)
        self._av_area_color_dst = self._preset_color_dst
        # Elections
        self.election_costs = election_costs
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
        self.search_pairs = list(combinations(range(0, self.options.shape[0]), 2))
        self.option_vec = np.arange(self.options.shape[0])  # Also to speed up
        self.color_search_pairs = list(combinations(range(0, num_colors), 2))
        # Create color cells (IDs start after areas+agents)
        self.color_cells: List[Optional[ColorCell]] = [None] * (height * width)  # TODO change to using mesas AgentSet class!
        self._initialize_color_cells(id_start=num_agents + num_areas)
        # Create voting agents (IDs start after areas)
        self.voting_agents: List[Optional[VoteAgent]] = [None] * num_agents    # TODO change to using mesas AgentSet class!
        self.personalities = self.create_personalities(num_personalities)
        self.personality_distribution = ParticipationModel.pers_dist(num_personalities, rng=self.np_random)
        self.initialize_voting_agents(id_start=num_areas)
        # Area variables
        self.global_area = self.initialize_global_area()
        self.areas: List[Optional[Area]] = [None] * num_areas    # TODO change to using mesas AgentSet class!
        self.av_area_height = av_area_height
        self.av_area_width = av_area_width
        self.area_size_variance = area_size_variance
        # Adjust the color pattern to make it less random (see color patches)
        self.adjust_color_pattern(color_patches_steps, patch_power)
        # Create areas
        self.initialize_all_areas()
        # Data collector
        # TODO: NOTE: I think I should (consider) set up only areas as mesa Agents and everything else as classes
        #   because the datacollector goes through all the agents (cells, voting agents) even though we only need to go through the areas.
        self.datacollector = self.initialize_datacollector()
        # Collect initial data
        self.datacollector.collect(self)

    @property
    def height(self) -> int:
        return self.grid.height

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def num_colors(self) -> int:
        return len(self.colors)

    @property
    def av_area_color_dst(self) -> np.ndarray:
        return self._av_area_color_dst

    @av_area_color_dst.setter
    def av_area_color_dst(self, value) -> None:
        self._av_area_color_dst = value

    @property
    def num_agents(self) -> int:
        return len(self.voting_agents)

    @property
    def num_areas(self) -> int:
        return len(self.areas)

    @property
    def preset_color_dst(self) -> np.ndarray:
        return self._preset_color_dst

    def _initialize_color_cells(self, id_start=0) -> None:
        """
        Initialize one ColorCell per grid cell.
        Args:
            id_start (int): The starting ID to ensure unique IDs.
        """
        # Create a color cell for each cell in the grid
        for idx, (_, (col, row)) in enumerate(self.grid.coord_iter()):
            # Assign unique ID after areas and agents
            unique_id = id_start + idx
            # The colors are chosen by a predefined color distribution
            color = self.color_by_dst(self._preset_color_dst)
            # Create the cell (skip ids for area and voting agents)
            cell = ColorCell(unique_id, self, (col, row), color)
            # Add to the 'model.color_cells' list (for faster access)
            self.color_cells[idx] = cell  # TODO: change to using the grid(?)

    def initialize_voting_agents(self, id_start=0) -> None:
        """
        This method initializes as many voting agents as set in the model with
        a randomly chosen personality. It places them randomly on the grid.
        It also ensures that each agent is assigned to the color cell it is
        standing on.
        Args:
            id_start (int): The starting ID for agents to ensure unique IDs.
        """
        # Testing parameter validity
        if self.num_agents < 1:
            raise ValueError("The number of agents must be at least 1.")
        dist = self.personality_distribution
        assets = self.common_assets // self.num_agents
        nr = len(self.personalities)
        for idx in range(self.num_agents):
            # Assign unique ID after areas
            unique_id = id_start + idx
            # Get a random position
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            # Choose a personality based on the distribution
            personality_idx = self.np_random.choice(nr, p=dist)
            personality = self.personalities[personality_idx]
            # Create agent without appending (add to the pre-defined list)
            agent = VoteAgent(unique_id, self, (x, y), personality,
                              personality_idx, assets=assets, add=False)
            self.voting_agents[idx] = agent  # Add using the index (faster)
            # Add the agent to the grid by placing it on a ColorCell
            cell = self.grid.get_cell_list_contents([(x, y)])[0]
            if TYPE_CHECKING:
                cell = cast(ColorCell, cell)
            cell.add_agent(agent)

    def init_color_probs(self, election_impact) -> np.ndarray:
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

    def initialize_area(self, a_id: int, x_coord, y_coord) -> None:
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

        Initializes `num_areas` and places them directly on the grid.
        But if `self.num_areas == 0`, the method exits early.

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
        nr_areas_x = self.grid.width // self.av_area_width
        nr_areas_y = self.grid.height // self.av_area_height
        # Calculate the distance between the areas
        area_x_dist = self.grid.width // nr_areas_x
        area_y_dist = self.grid.height // nr_areas_y
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


    def initialize_global_area(self) -> Area:
        """
        Initializes the global area spanning the whole grid.

        Returns:
            Area: The global area (with unique_id set to -1 and idx to (0, 0)).
        """
        global_area = Area(-1, self, self.height, self.width, 0)
        # Place the area in the grid using its indexing field
        # this adds the corresponding color cells and voting agents to the area
        global_area.idx_field = (0, 0)
        return global_area


    def create_personalities(self, n: int) -> np.ndarray:
        """
        Creates n unique personalities as permutations of color indices.

        Args:
            n (int): Number of unique personalities.

        Returns:
            np.ndarray: Shape `(n, num_colors)`.

        Raises:
            ValueError: If `n` exceeds the possible unique permutations.

        Example:
            for n=2 and self.num_colors=3, the function could return:

            [[1, 0, 2],
            [2, 1, 0]]
        """
        # p_colors = range(1, self.num_colors)  # Personalities exclude white
        n_colors = self.num_colors
        max_permutations = factorial(n_colors)
        if n > max_permutations or n < 1:
            raise ValueError(f"Cannot generate {n} unique personalities: "
                             f"only {max_permutations} unique ones exist.")
        selected_permutations = set()
        while len(selected_permutations) < n:
            # Sample a permutation lazily and add it to the set
            perm = tuple(self.random.sample(range(n_colors), n_colors))
            selected_permutations.add(perm)

        return np.array(list(selected_permutations))


    def initialize_datacollector(self) -> mesa.DataCollector:
        color_data = {f"Color {i}": get_color_distribution_function(i) for i in
                      range(self.num_colors)}
        return mesa.DataCollector(
            model_reporters={
                "Collective assets": compute_collective_assets,
                "Gini Index (0-100)": compute_gini_index,
                "Voter turnout globally (in percent)": get_voter_turnout,
                **color_data,
                "GridColors": get_grid_colors
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
                "GiniIndex": get_area_gini_index,
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
        # Early exit if step limit reached
        if self.max_steps is not None and self.scheduler.steps >= self.max_steps:
            self.running = False
            return
        # Conduct elections in the areas
        # and then mutate the color cells according to election outcomes
        self.scheduler.step()
        # Update the global color distribution
        self.update_av_area_color_dst()
        # Collect data for monitoring and data analysis
        self.datacollector.collect(self)
        # Enforce step limit after step executed
        if self.max_steps is not None and self.scheduler.steps >= self.max_steps:
            self.running = False

    def adjust_color_pattern(self, color_patches_steps: int, patch_power: float):
        """Adjusting the color pattern to make it less predictable.

        Args:
            color_patches_steps: How often to run the color-patches step.
            patch_power: The power of the patching (like a radius of impact).
        """
        cells = self.color_cells
        for _ in range(color_patches_steps):
            # print(f"Color adjustment step {_}")
            self.random.shuffle(cells)
            for cell in cells:
                most_common_color = self.color_patches(cell, patch_power)
                cell.color = most_common_color


    def create_color_distribution(self, heterogeneity: float) -> np.ndarray:
        """
        Create a normalized color distribution biased by the heterogeneity factor.

        Args:
            heterogeneity (float): Standard deviation for Gaussian sampling.
        """
        # Vectorized sampling: mean=1, std=heterogeneity, shape=(num_colors,)
        values = np.abs(
            self.np_random.normal(1.0, heterogeneity, self.num_colors))
        # Normalize
        values /= values.sum()
        return values

    def color_patches(self, cell: ColorCell, patch_power: float) -> int:
        """
        Meant to create a less random initial color distribution
        using a similar logic to the color patches model.
        It uses a (normalized) bias coordinate to center the impact of the
        color patches structures impact around.

        Args:
            cell (ColorCell): The cell possibly changing color.
            patch_power (float): Radius-like impact around bias point.

        Returns:
            int: Consensus color or the cell's own color if no consensus.
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
        neighbor_cells = self.grid.get_neighbors((cell.col, cell.row),
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

    @ staticmethod
    def pers_dist(size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Create a normalized non-negative distribution of length `size`.
        Generates a sorted absolute normal sample and normalizes to sum to one.
        """
        if rng is None:
            rng = np.random.default_rng()
        dist = rng.normal(0, 1, size)
        dist.sort()
        dist = np.abs(dist)
        total = dist.sum()
        if total == 0:
            # Edge-case: all zeros; fallback to uniform
            return np.full(size, 1.0 / size)
        return dist / total

    @staticmethod
    def create_all_options(n: int, include_ties=False) -> np.ndarray:
        """
        Creates a matrix (an array of all possible ranking vectors),
        if specified including ties.
        Rank values start from 0.

        Args:
            n (int): The number of items to rank (number of colors in our case)
            include_ties (bool): If True, rankings include ties.

        Returns:
            np.ndarray: A matrix containing all possible ranking vectors.
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
    def color_by_dst(color_distribution: np.ndarray) -> int:
        """
        Selects a color (int) based on the given color_distribution array,
        where each entry represents the probability of selecting that index.

        Args:
            color_distribution: Array determining the selection probabilities.

        Returns:
            int: The selected index based on the given probabilities.

        Raises:
            ValueError: If probabilities do not sum to 1 or contain negatives.

        Example:
            color_distribution = [0.2, 0.3, 0.5]
            Color 1 will be selected with a probability of 0.3.
        """
        if abs(sum(color_distribution) -1) > 1e-8:
            raise ValueError("The color_distribution array must sum to 1.")
        r = np.random.random()  # Float between 0 and 1
        cumulative_sum = 0.0
        for color_idx, prob in enumerate(color_distribution):
            if prob < 0:
                raise ValueError("color_distribution contains negative value.")
            cumulative_sum += prob
            if r < cumulative_sum:  # Compare r against the cumulative probability
                return color_idx

        # This point should never be reached.
        raise ValueError("Unexpected error in color_distribution.")


def get_color_distribution_function(color: int) -> Callable[
    [ParticipationModel], float]:
    """
    Returns a lambda to extract a single color's distribution from the model.

    Args:
        color (int): Index of the color.

    Returns:
        Callable[[ParticipationModel], float]: Extractor.
    """
    return lambda m: float(m.av_area_color_dst[color])


def get_area_voter_turnout(area: Area) -> Optional[float]:
    return area.voter_turnout if isinstance(area, Area) else None


def get_area_dist_to_reality(area: Area) -> Optional[float]:
    return area.dist_to_reality if isinstance(area, Area) else None


def get_area_color_distribution(area: Area) -> Optional[list[float]]:
    return area.color_distribution.tolist() if isinstance(area, Area) else None


def get_election_results(area: Area) -> Optional[list[int]]:
    """
    Returns the voted ordering as a list or None if not available.

    Returns:
        list[int] | None
    """
    if isinstance(area, Area) and area.voted_ordering is not None:
        return area.voted_ordering.tolist()
    return None


# def get_area_personality_based_reward(area: Area) -> Optional[float]:
#     return area.personality_based_reward if isinstance(area, Area) else None


def get_area_gini_index(area: Area) -> Optional[float]:
    """Per-area Gini index (0-100) computed from agents' assets.
    """
    if not isinstance(area, Area):
        return None
    assets = [a.assets for a in area.agents]
    return float(gini_index_0_100(assets))
