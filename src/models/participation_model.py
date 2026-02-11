from typing import TYPE_CHECKING, cast, List, Optional, Callable
import mesa
import numpy as np
from math import factorial
from src.agents import Area, VoteAgent, ColorCell
from src.utils.social_welfare_functions import majority_rule, approval_voting
from src.utils.distance_functions import spearman_fr_order, kendall_tau_order
from itertools import permutations, product, combinations
from src.utils.metrics import (compute_gini_index, compute_collective_assets,
                               get_voter_turnout, get_grid_colors,
                               gini_index_0_100)
from src.utils.rng import (
    set_seed,
    np_rng,
    np_rng_viz,
    np_rng_debug,
    py_rng,
    py_rng_viz,
    py_rng_debug,
)


# Voting rules to be accessible by index
social_welfare_functions = [majority_rule, approval_voting]
# Distance functions
# (explicitly ordering-based)
distance_functions = [spearman_fr_order, kendall_tau_order]


class CustomScheduler(mesa.time.BaseScheduler):
    def step(self):
        """
        Execute the step function for all area-agents.
        """
        model = self.model
        if TYPE_CHECKING:
            model = cast(ParticipationModel, model)
        # TODO: Logg step 0 as initial state
        self.steps += 1
        self.time += 1
        # Step through Area agents (in "random" order)
        model.random.shuffle(model.areas)
        # Mutation happens before stepping.
        # Before the first step, no election has taken place, then no mutation.
        if self.steps > 1:
            for area in model.areas:
                # Mutation applies changes of the last election (previous step).
                area.mutate_cells()
        if not model.no_overlap:  # There may be overlap
            for area in model.areas:
                area.update_color_distribution()
        # Update color distribution (colrs may have mutated)
        model.update_global_color_distribution()
        for area in model.areas:
            area.step()
        # TODO: add global election?


    @property
    def agents(self):
        model = self.model
        if TYPE_CHECKING:
            model = cast(ParticipationModel, model)
        # Return all area agents
        return model.areas

    @property
    def global_area(self) -> Area:
        model = self.model
        if TYPE_CHECKING:
            model = cast(ParticipationModel, model)
        return model.global_area


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
            takes agent score vectors as input and returns an ordering.
        distance_func (Callable): A function used to calculate a
            distance metric when comparing orderings. It takes two orderings
            and returns a numeric distance score.
        mu (float): Mutation rate; the probability of each color cell to mutate
            after an elections.
        color_probs (ndarray):
            Probabilities used to determine individual color mutation outcomes.
        options (ndarray): Matrix where each row is an option ordering
            (permutation) available to agents.
        option_vec (ndarray): Array holding the indices of the available options
            for computational efficiency.
        color_cells (list[ColorCell]): List of all color cells.
            Initialized during the model setup.
        voting_agents (list[VoteAgent]): List of all voting agents.
            Initialized during the model setup.
        personality_groups (list): List of personality groups available for agents.
        personality_group_distribution (ndarray): The (global) probability
            distribution of personality groups among all agents.
        areas (list[Area]): List of areas (regions or territories within the
            grid) in which elections take place. Initialized during model setup.
        global_area (Area): The area encompassing the entire grid.
        av_area_height (int): Average height of areas in the simulation.
        av_area_width (int): Average width of areas created in the simulation.
        area_size_variance (float): Variance in area sizes to introduce
            non-uniformity among election territories.
        common_assets (float): Total resources to be distributed among all agents.
        av_area_color_dst (ndarray): Current (area)-average color distribution.
        global_color_dst (ndarray): Current global color distribution across the grid.
        election_cost_rate (float): Cost/effort associated with participating in elections (relative to assets).
        known_cells (int): Number of cells each agent knows the color of.
        datacollector (mesa.DataCollector): A tool for collecting data
            (metrics and statistics) at each simulation step.
        scheduler (CustomScheduler): The scheduler responsible for executing the
            step function.
        _preset_color_dst (ndarray): A predefined global color distribution
            (set randomly) that affects cell initialization globally.
        _no_overlap (bool): A flag indicating areas don't overlap. Speeds up certain computations if True.
    """

    def __init__(
        self,
        height,
        width,
        num_agents,
        num_colors,
        num_personality_groups,
        mu,
        election_impact_on_mutation,
        known_cells,
        num_areas,
        av_area_height,
        av_area_width,
        area_size_variance,
        patch_power,
        color_patches_steps,
        heterogeneity,
        rule_idx,
        distance_idx,
        election_cost_rate,
        reward_rate_common: float = 0.0,
        reward_rate_personal: float = 0.0,
        reward_threshold_common: float = 0.5,
        reward_threshold_personal: float = 0.5,
        abstention_share: float = 1.0,
        seed=None,
        max_steps: Optional[int] = None,
        participation_alpha: float = 0.05,
        participation_beta: float = 1.0,
        participation_init_q: float = 0.0,
        participation_q_max: float = 50.0,
        bias_toward_participation: float = 0.0,
        participation_baseline_alpha: float = 0.1,
        altruism_alpha: float = 0.05,
        altruism_init: float = 0.5,
        altruism_clip_min: float = 0.0,
        altruism_clip_max: float = 1.0,
        altruism_learning: bool = False,
        altruism_static: float = 0.5,
        satisfaction_mode: str = "area",  # "global", "area", "knowledge", or "combination"
        satisfaction_baseline_alpha: float = 0.1,
        personal_opt_dist_concentration: float = 1.0,
        common_assets=None
    ):
        super().__init__()
        self._seed = seed
        # Store scalar params early because agent init depends on them.
        self.known_cells = known_cells  # Integer
        # Adaptive participation learning parameters (global per agent)
        self.participation_alpha = float(participation_alpha)  # Learning rate
        self.participation_beta = float(participation_beta)  # Sensitivity
        self.participation_init_q = float(participation_init_q)
        self.participation_q_max = float(participation_q_max)
        self.bias_toward_participation = float(bias_toward_participation)
        self.participation_baseline_alpha = float(participation_baseline_alpha)
        if not (0.0 <= self.participation_baseline_alpha <= 1.0):
            raise ValueError("participation_baseline_alpha must be in [0,1].")
        # Adaptive altruism learning parameters (global per agent)
        self.altruism_alpha = float(altruism_alpha)  # Learning rate. How fast q changes in response to the signal.
        self.altruism_init = float(altruism_init)
        self.altruism_clip_min = float(altruism_clip_min)
        self.altruism_clip_max = float(altruism_clip_max)
        self.altruism_learning = bool(altruism_learning)
        self.altruism_static = float(altruism_static)
        if not (0.0 <= self.altruism_static <= 1.0):
            raise ValueError("altruism_static must be in [0,1].")
        self.satisfaction_mode = str(satisfaction_mode)
        if self.satisfaction_mode not in {"global", "area", "knowledge", "combination"}:
            raise ValueError(
                "satisfaction_mode must be one of: global, area, knowledge, combination."
            )
        self.satisfaction_baseline_alpha = float(satisfaction_baseline_alpha)
        if not (0.0 <= self.satisfaction_baseline_alpha <= 1.0):
            raise ValueError("satisfaction_baseline_alpha must be in [0,1].")
        self.personal_opt_dist_concentration = personal_opt_dist_concentration

        # Initialize RNGs early (centralized)
        set_seed(seed)
        self.np_random = np_rng()
        self.random = py_rng()
        # Dedicated streams for visualization/debug to avoid perturbing simulation RNG.
        self.rng_viz = np_rng_viz()
        self.rng_debug = np_rng_debug()
        self.random_viz = py_rng_viz()
        self.random_debug = py_rng_debug()
        if seed is not None:
            print(f"Set models random seed to {seed}")

        # Step control
        self.max_steps: Optional[int] = max_steps
        self.running: bool = True
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
        self._av_area_color_dst = self._preset_color_dst.copy()  # TODO: Deal with overlaps and size diffs
        self.global_color_dst = self._preset_color_dst.copy()
        # Elections
        self.election_cost_rate = election_cost_rate
        # Reward scaling knobs
        self.reward_rate_common = float(reward_rate_common)
        self.reward_rate_personal = float(reward_rate_personal)
        self.reward_threshold_common = float(reward_threshold_common)
        self.reward_threshold_personal = float(reward_threshold_personal)
        self.abstention_share = max(0.0, min(1.0, float(abstention_share)))

        # Wrap voting rules so they use deterministic RNG
        # Keep self.voting_rule as the base function for tests.
        self.voting_rule = social_welfare_functions[rule_idx]
        self.voting_rng = self.np_random
        self.distance_func = distance_functions[distance_idx]
        self.options = self.create_all_options(num_colors)
        # Simulation variables
        self.mu = mu  # Mutation rate for the color cells (0.1 = 10 % mutate)
        self.common_assets = 100*num_agents if common_assets is None else common_assets
        # Election impact factor on color mutation through a probability array
        self.color_probs = self.init_color_probs(election_impact_on_mutation)
        # Create search pairs once for faster iterations when comparing orderings
        # (Removed unused self.search_pairs to avoid O(options^2) memory growth.)
        self.option_vec = np.arange(self.options.shape[0])  # Also to speed up
        self.color_search_pairs = list(combinations(range(0, num_colors), 2))
        # Create color cells (IDs start after areas+agents)
        self.color_cells: List[Optional[ColorCell]] = [None] * (height * width)  # TODO change to using mesas AgentSet class!
        self._initialize_color_cells(id_start=num_agents + num_areas)
        # Create voting agents (IDs start after areas)
        self.voting_agents: List[Optional[VoteAgent]] = [None] * num_agents    # TODO change to using mesas AgentSet class!
        self.personality_groups = self.create_personality_groups(num_personality_groups)
        pg_dst = ParticipationModel.pers_dist(num_personality_groups, rng=self.np_random)
        self.initialize_voting_agents(intended_dst=pg_dst, id_start=num_areas)
        self.personality_group_distribution = self._initialize_personality_group_distribution()  # Static
        # Area variables
        self.global_area = self.initialize_global_area()
        self.areas: List[Optional[Area]] = [None] * num_areas    # TODO change to using mesas AgentSet class!
        self.av_area_height = av_area_height
        self.av_area_width = av_area_width
        self.area_size_variance = area_size_variance
        self._no_overlap = False  # True if areas are instantiated without overlap (speeds up things)
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
    def num_agents(self) -> int:
        return len(self.voting_agents)

    @property
    def num_areas(self) -> int:
        return len(self.areas)

    @property
    def preset_color_dst(self) -> np.ndarray:
        return self._preset_color_dst

    @property
    def av_area_color_dst(self) -> np.ndarray:
        return self._av_area_color_dst

    @av_area_color_dst.setter
    def av_area_color_dst(self, value) -> None:
        self._av_area_color_dst = value

    @property
    def no_overlap(self) -> bool:
        return self._no_overlap

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
            color = self.color_by_dst_rng(self._preset_color_dst)
            # Create the cell (skip ids for area and voting agents)
            cell = ColorCell(unique_id, self, (col, row), color)
            # Add to the 'model.color_cells' list (for faster access)
            self.color_cells[idx] = cell  # TODO: change to using the grid(?)

    def initialize_voting_agents(self, intended_dst, id_start = 0) -> None:
        """
        This method initializes as many voting agents as set in the model with
        a randomly chosen personality_group. It places them randomly on the grid.
        It also ensures that each agent is assigned to the color cell it is
        standing on.
        Args:
            id_start (int): The starting ID for agents to ensure unique IDs.
            intended_dst (np.ndarray): The intended distribution of personality groups.
        """
        # Testing parameter validity
        if self.num_agents < 1:
            raise ValueError("The number of agents must be at least 1.")
        assets = self.common_assets / self.num_agents  # TODO: always equal dist?
        nr = len(self.personality_groups)
        for idx in range(self.num_agents):
            # Assign unique ID after areas
            unique_id = id_start + idx
            # Get a random position
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            # Choose a personality_group based on the distribution
            personality_group_idx = self.np_random.choice(nr, p=intended_dst)
            personality_group = self.personality_groups[personality_group_idx]
            # Create agent without appending (add to the pre-defined list)
            agent = VoteAgent(unique_id, self, (x, y), personality_group,
                              personality_group_idx, assets=assets, add=False)
            self.voting_agents[idx] = agent  # Add using the index (faster)
            # Add the agent to the grid by placing it on a ColorCell
            cell = self.grid.get_cell_list_contents([(x, y)])[0]
            if TYPE_CHECKING:
                cell = cast(ColorCell, cell)
            cell.add_agent(agent)

    def _initialize_personality_group_distribution(self) -> np.ndarray:
        counts = np.bincount(
            [a.personality_group_idx for a in self.voting_agents],
            minlength=len(self.personality_groups))
        return counts / counts.sum()

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
        self._no_overlap = (
            self.area_size_variance == 0
            and self.av_area_width > 0
            and self.av_area_height > 0
            and self.width % self.av_area_width == 0
            and self.height % self.av_area_height == 0
            and self.num_areas == nr_areas_x * nr_areas_y
        )
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
        if missing > 0:
            self._no_overlap = False
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


    def create_personality_groups(self, n: int) -> np.ndarray:
        """
        Creates n unique personality_groups as permutations of color indices.

        Args:
            n (int): Number of unique personality_groups.

        Returns:
            np.ndarray: Shape `(n, num_colors)`.

        Raises:
            ValueError: If `n` exceeds the possible unique permutations.

        Example:
            for n=2 and self.num_colors=3, the function could return:

            [[1, 0, 2],
            [2, 1, 0]]
        """
        n_colors = self.num_colors
        max_permutations = factorial(n_colors)
        if n > max_permutations or n < 1:
            raise ValueError(f"Cannot generate {n} unique personality_groups: "
                             f"only {max_permutations} unique ones exist.")
        selected_permutations = set()
        while len(selected_permutations) < n:
            # Sample a permutation lazily and add it to the set
            perm = tuple(self.random.sample(range(n_colors), n_colors))
            selected_permutations.add(perm)

        return np.array(list(selected_permutations))


    def initialize_datacollector(self) -> mesa.DataCollector:
        # Live (run.py) visualization expects snake_case keys.
        color_data = {f"color_{i}": get_color_distribution_function(i) for i in range(self.num_colors)}

        def mean_p_participation(m: "ParticipationModel") -> float:
            agents = m.voting_agents
            if not agents:
                return 0.0
            vals = [float(a.participation_probability()) for a in agents if a is not None]
            return float(np.mean(vals)) if vals else 0.0

        def mean_altruism(m: "ParticipationModel") -> float:
            agents = m.voting_agents
            if not agents:
                return 0.0
            vals = [float(a.altruism_factor) for a in agents if a is not None]
            return float(np.mean(vals)) if vals else 0.0
        def mean_satisfaction(m: "ParticipationModel") -> float:
            agents = m.voting_agents
            if not agents:
                return 0.0
            vals = [float(a.satisfaction_value) for a in agents if a is not None]
            return float(np.mean(vals)) if vals else 0.0

        return mesa.DataCollector(
            model_reporters={
                "collective_assets": compute_collective_assets,
                "gini_index": compute_gini_index,
                "turnout": get_voter_turnout,
                "mean_p_participation": mean_p_participation,
                "mean_altruism": mean_altruism,
                "mean_satisfaction": mean_satisfaction,
                **color_data,
                "grid_colors": get_grid_colors,
            },
            agent_reporters={
                # These are collected for all Mesa agents, but only Area agents return values.
                "turnout": get_area_voter_turnout,
                "dist_to_reality": get_area_dist_to_reality,
                "area_color_distribution": get_area_color_distribution,
                "elected_color": get_election_results,
                "gini_index": get_area_gini_index,
            },
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
        # Collect data for monitoring and data analysis (pre-mutation).
        self.datacollector.collect(self)
        # Enforce step limit after step executed
        if self.max_steps is not None and self.scheduler.steps >= self.max_steps:
            # TODO: Apply one final mutation round (from previous election)
            #for area in self.areas:
            #    area.mutate_cells()
            #self.update_global_color_distribution()
            # TODO: logg final state after mutation
            #self.datacollector.collect(self)
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
            return self.color_by_dst_rng(self._preset_color_dst)
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


    def update_av_area_color_dst(self) -> np.ndarray:
        """
        This method updates the av_area_color_dst attribute of the model.
        Beware: Overlaps and size difference of areas is not currently accounted for,
        so this is a simple average across areas meant only for non-overlapping,
        equally sized area distributions.
        """
        sums = np.zeros(self.num_colors)
        for area in self.areas:
            if area.unique_id != -1:  # Exclude global area
                sums += area.color_distribution
        # Return the average color distributions
        self.av_area_color_dst = sums / self.num_areas
        return self.av_area_color_dst


    def update_global_color_distribution(self) -> None:
        """
        This method updates the global color distribution based on the current
        state of the grid. It calculates the distribution of colors across all
        color cells and normalizes it to sum to 1.
        """
        if self.area_size_variance == 0 and self._no_overlap:
            self.global_color_dst = self.update_av_area_color_dst()
            return
        elif self.width * self.height > 1e+5:
            print("Warning: Updating global color distribution on large grids may be slow.")
        color_counts = np.zeros(self.num_colors)
        for cell in self.color_cells:
            color_counts[cell.color] += 1
        total_cells = len(self.color_cells)
        if total_cells > 0:
            self.global_color_dst = color_counts / total_cells

    @staticmethod
    def pers_dist(size: int, *, rng: np.random.Generator) -> np.ndarray:
        """
        Create a normalized non-negative distribution of length `size`.
        Generates a sorted absolute normal sample and normalizes to sum to one.
        """
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
        Creates a matrix (an array) of all possible orderings (permutations),
        optionally including ties (rank vectors).
        Rank values start from 0.

        Args:
            n (int): The number of items to rank (number of colors in our case)
            include_ties (bool): If True, include rank vectors with ties.

        Returns:
            np.ndarray: A matrix containing all possible orderings or rank vectors.
        """
        if include_ties:
            # Create all possible combinations and sort out invalid rank vectors
            # i.e. [1, 1, 1] or [1, 2, 2] aren't valid as no option is ranked first.
            r = np.array([np.array(comb) for comb in product(range(n), repeat=n)
                          if set(range(max(comb))).issubset(comb)])
        else:
            r = np.array([np.array(p) for p in permutations(range(n))])
        return r

    def color_by_dst_rng(self, color_distribution: np.ndarray) -> int:
        """Deterministic sampling using the model's seeded RNG."""
        if abs(sum(color_distribution) -1) > 1e-8:
            raise ValueError("The color_distribution array must sum to 1.")
        r = float(self.np_random.random())
        cumulative_sum = 0.0
        for color_idx, prob in enumerate(color_distribution):
            if prob < 0:
                raise ValueError("color_distribution contains negative value.")
            cumulative_sum += prob
            if r < cumulative_sum:
                return int(color_idx)
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
    return lambda m: float(m.global_color_dst[color])


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


def get_area_gini_index(area: Area) -> Optional[float]:
    """Per-area Gini index (0-100) computed from agents' assets.
    """
    if not isinstance(area, Area):
        return None
    assets = [a.assets for a in area.agents]
    return float(gini_index_0_100(assets))
