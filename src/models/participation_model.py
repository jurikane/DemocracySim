from typing import TYPE_CHECKING, cast, List, Optional, Callable
import mesa
import numpy as np
from math import factorial
from src.agents import Area, VoteAgent, ColorCell
from src.utils.social_welfare_functions import (majority_rule, approval_voting,
                                                utilitarian_rule, borda_rule)
from src.utils.distance_functions import spearman_fr_order, kendall_tau_order
from itertools import permutations, product, combinations
from src.utils.metrics import (compute_gini_index, compute_collective_assets,
                               get_voter_turnout, get_grid_colors)
from src.utils.helpers import (get_area_voter_turnout, is_rate_btw_0_and_1,
                                get_area_dist_to_reality, get_election_results,
                                get_area_color_distribution, get_area_gini_index,
                                is_learning_rate, ensure_rate_0_1, ensure_choice,
                                ensure_int_ge_0, ensure_finite_ge_0, ensure_finite_gt_0)
from src.utils.rng import (
    set_seed,
    np_rng,
    np_rng_viz,
    np_rng_debug,
    np_rng_participation,
    np_rng_voting,
    py_rng,
    py_rng_viz,
    py_rng_debug,
)

# Voting rules to be accessible by index
social_welfare_functions = [majority_rule, approval_voting, utilitarian_rule, borda_rule]
social_welfare_function_short_names = ["Majority", "Approval", "Utilitarian", "Borda"]
# Distance functions
# (explicitly ordering-based)
distance_functions = [spearman_fr_order, kendall_tau_order]
distance_function_short_names = ["SpearmanFootrule", "KendallTau"]


class CustomScheduler(mesa.time.BaseScheduler):
    def step(self):
        """
        Execute the step function for all area-agents.
        """
        model = self.model
        if TYPE_CHECKING:
            model = cast(ParticipationModel, model)
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
        initial_agent_assets (float): Initial assets assigned to each agent.
        _av_area_color_dst (ndarray): Current (area)-average color distribution.
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

    # -----------------
    # Validation / setup helpers
    # -----------------
    @staticmethod
    def _validate_color_and_personality_space(
        *,
        num_colors,
        num_personality_groups,
    ) -> tuple[int, int]:
        """Validate color-domain size and personality-group cardinality contracts."""
        if not isinstance(num_colors, int):
            raise ValueError(f"num_colors must be int, got {type(num_colors)}")
        if num_colors < 2:
            raise ValueError("num_colors must be >= 2.")

        n_groups = ensure_int_ge_0("num_personality_groups", num_personality_groups)
        if n_groups < 1:
            raise ValueError("num_personality_groups must be >= 1.")

        max_personality_groups = factorial(int(num_colors))
        if n_groups > max_personality_groups:
            raise ValueError(
                f"num_personality_groups={n_groups} exceeds "
                f"max unique permutations {max_personality_groups} for num_colors={num_colors}."
            )

        # Options are all permutations of colors; this grows as num_colors!.
        # Keep a conservative cap to avoid accidentally creating enormous option spaces.
        max_options = factorial(int(num_colors))
        if max_options > 50_000:
            raise ValueError(
                f"num_colors={num_colors} implies {max_options} options (num_colors!), "
                "which is too large for this simulation configuration. "
                "Reduce num_colors (e.g. <= 8) or implement an alternative option representation."
            )
        return int(num_colors), int(n_groups)

    @staticmethod
    def _validate_population_topology_inputs(
        *,
        num_agents,
        num_areas,
        height,
        width,
        av_area_height,
        av_area_width,
        area_size_variance,
    ) -> tuple[int, int, int, int, float]:
        """Validate agent/area counts and geometry-related scalar inputs."""
        n_agents = ensure_int_ge_0("num_agents", num_agents)
        if n_agents < 1:
            raise ValueError("num_agents must be >= 1.")

        n_areas = ensure_int_ge_0("num_areas", num_areas)
        if n_areas < 1:
            raise ValueError("num_areas must be >= 1.")
        if n_areas > int(height) * int(width):
            raise ValueError(
                f"num_areas={n_areas} exceeds available grid anchor slots "
                f"({int(height) * int(width)} for {width}x{height})."
            )

        av_h = ensure_int_ge_0("av_area_height", av_area_height)
        av_w = ensure_int_ge_0("av_area_width", av_area_width)
        if av_h == 0 or av_w == 0:
            raise ValueError("av_area_height and av_area_width must be >= 1.")
        if av_h > int(height):
            raise ValueError(f"av_area_height={av_h} exceeds grid height={height}.")
        if av_w > int(width):
            raise ValueError(f"av_area_width={av_w} exceeds grid width={width}.")

        area_var = ensure_finite_ge_0("area_size_variance", area_size_variance)
        if area_var > 1.0:
            raise ValueError("area_size_variance must be in [0,1].")
        return int(n_agents), int(n_areas), int(av_h), int(av_w), float(area_var)

    def _configure_participation_learning(
        self,
        *,
        participation_alpha,
        participation_beta,
        participation_init_q,
        participation_q_max,
        bias_toward_participation,
        participation_baseline_alpha,
    ) -> None:
        """Validate and assign participation-learning knobs."""
        self.participation_alpha = is_learning_rate(participation_alpha)

        self.participation_beta = float(participation_beta)
        if not np.isfinite(self.participation_beta) or self.participation_beta < 0.0:
            raise ValueError("participation_beta must be finite and >= 0.")

        self.participation_init_q = float(participation_init_q)
        if not np.isfinite(self.participation_init_q):
            raise ValueError("participation_init_q must be finite.")

        self.participation_q_max = float(participation_q_max)
        if not np.isfinite(self.participation_q_max) or self.participation_q_max < 0.0:
            raise ValueError("participation_q_max must be finite and >= 0.")

        self.bias_toward_participation = float(bias_toward_participation)
        if not np.isfinite(self.bias_toward_participation) or not (-1.0 <= self.bias_toward_participation <= 1.0):
            raise ValueError("bias_toward_participation must be finite and in [-1,1].")

        self.participation_baseline_alpha = ensure_rate_0_1(
            "participation_baseline_alpha", participation_baseline_alpha
        )

    def _configure_altruism_learning_and_satisfaction(
        self,
        *,
        altruism_alpha,
        altruism_init,
        altruism_clip_min,
        altruism_clip_max,
        altruism_learning,
        altruism_static,
        satisfaction_mode,
        satisfaction_baseline_alpha,
    ) -> None:
        """Validate and assign altruism-learning and satisfaction knobs."""
        self.altruism_alpha = is_learning_rate(altruism_alpha)

        self.altruism_init = float(altruism_init)
        if not np.isfinite(self.altruism_init) or not (0.0 <= self.altruism_init <= 1.0):
            raise ValueError("altruism_init must be finite and in [0,1].")

        self.altruism_clip_min = float(altruism_clip_min)
        self.altruism_clip_max = float(altruism_clip_max)
        if (
            (not np.isfinite(self.altruism_clip_min))
            or (not np.isfinite(self.altruism_clip_max))
            or (self.altruism_clip_min > self.altruism_clip_max)
        ):
            raise ValueError("altruism_clip_min/max must be finite and satisfy clip_min <= clip_max.")

        self.altruism_learning = bool(altruism_learning)
        self.altruism_static = ensure_rate_0_1("altruism_static", altruism_static)

        self.satisfaction_mode = ensure_choice(
            "satisfaction_mode",
            str(satisfaction_mode),
            {"global", "area", "knowledge", "combination"},
        )
        self.satisfaction_baseline_alpha = ensure_rate_0_1(
            "satisfaction_baseline_alpha", satisfaction_baseline_alpha
        )

    def _configure_rules_rewards_and_distance(
        self,
        *,
        rule_idx,
        distance_idx,
        election_cost_rate,
        reward_rate_common,
        reward_rate_personal,
        break_even_distance_common,
        break_even_distance_personal,
        abstention_share,
        num_colors: int,
    ) -> None:
        """Validate and assign voting-rule, distance, and reward knobs."""
        vr, vr_names, vr_name, vr_i_names, vr_i_name = self._get_voting_rule_conf(rule_idx)
        self.rule_idx = rule_idx
        self.voting_rule = vr
        self.voting_rule_names = vr_names
        self.voting_rule_name = vr_name
        # Implementation names are stored alongside display names so runs can be
        # reproduced even if UI labels change.
        self.voting_rule_implementation_names = vr_i_names
        self.voting_rule_implementation_name = vr_i_name

        self.election_cost_rate = is_rate_btw_0_and_1(election_cost_rate)
        self.reward_rate_common = is_rate_btw_0_and_1(reward_rate_common)
        self.reward_rate_personal = is_rate_btw_0_and_1(reward_rate_personal)
        self.break_even_distance_common = is_rate_btw_0_and_1(break_even_distance_common)
        self.break_even_distance_personal = is_rate_btw_0_and_1(break_even_distance_personal)
        self.abstention_share = is_rate_btw_0_and_1(abstention_share)

        self.distance_idx = distance_idx
        dist, d_names, d_name, d_i_names, d_i_name = self._get_dist_conf(distance_idx)
        self.distance_func = dist
        self.distance_func_names = d_names
        self.distance_func_name = d_name
        self.distance_func_implementation_names = d_i_names
        self.distance_func_implementation_name = d_i_name
        self.options = self.create_all_options(num_colors)

    def _configure_environment_scalars(
        self,
        *,
        heterogeneity,
        mu,
        initial_agent_assets,
        election_impact_on_mutation,
        color_patches_steps,
        patch_power,
    ) -> None:
        """Validate and assign environment/economy scalar knobs."""
        self.heterogeneity = float(heterogeneity)
        if not np.isfinite(self.heterogeneity) or self.heterogeneity < 0.0:
            raise ValueError("heterogeneity must be finite and >= 0.")
        self._preset_color_dst = self.create_color_distribution(self.heterogeneity)
        self._av_area_color_dst = self._preset_color_dst.copy()
        self.global_color_dst = self._preset_color_dst.copy()

        self.mu = ensure_rate_0_1("mu", mu)
        self.initial_agent_assets = ensure_finite_ge_0("initial_agent_assets", initial_agent_assets)

        self.election_impact_on_mutation = float(election_impact_on_mutation)
        if not np.isfinite(self.election_impact_on_mutation) or self.election_impact_on_mutation < 0.0:
            raise ValueError("election_impact_on_mutation must be finite and >= 0.")
        self.color_probs = self.init_color_probs(self.election_impact_on_mutation)

        self.color_patches_steps = ensure_int_ge_0("color_patches_steps", color_patches_steps)
        self.patch_power = ensure_finite_ge_0("patch_power", patch_power)

    # -----------------
    # Initialization
    # -----------------
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
        break_even_distance_common: float = 0.5,
        break_even_distance_personal: float = 0.5,
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
        personal_preference_peakedness: float = 1.0,
        initial_agent_assets: float = 100.0
    ):
        super().__init__()
        self._seed = seed
        self._av_area_color_dst = np.asarray([], dtype=np.float64)
        self.global_color_dst = np.asarray([], dtype=np.float64)
        # --- Core sizing validation (fail-loud; avoids factorial explosions) ---
        num_colors, num_personality_groups = self._validate_color_and_personality_space(
            num_colors=num_colors,
            num_personality_groups=num_personality_groups,
        )
        num_agents, num_areas, av_h, av_w, area_var = self._validate_population_topology_inputs(
            num_agents=num_agents,
            num_areas=num_areas,
            height=height,
            width=width,
            av_area_height=av_area_height,
            av_area_width=av_area_width,
            area_size_variance=area_size_variance,
        )
        self.av_area_height = av_h
        self.av_area_width = av_w
        self.area_size_variance = area_var
        self.known_cells = ensure_int_ge_0("known_cells", known_cells)

        # Adaptive learning knobs
        self._configure_participation_learning(
            participation_alpha=participation_alpha,
            participation_beta=participation_beta,
            participation_init_q=participation_init_q,
            participation_q_max=participation_q_max,
            bias_toward_participation=bias_toward_participation,
            participation_baseline_alpha=participation_baseline_alpha,
        )
        self._configure_altruism_learning_and_satisfaction(
            altruism_alpha=altruism_alpha,
            altruism_init=altruism_init,
            altruism_clip_min=altruism_clip_min,
            altruism_clip_max=altruism_clip_max,
            altruism_learning=altruism_learning,
            altruism_static=altruism_static,
            satisfaction_mode=satisfaction_mode,
            satisfaction_baseline_alpha=satisfaction_baseline_alpha,
        )
        ppp = ensure_finite_gt_0("pp-peak", personal_preference_peakedness)
        self.personal_preference_peakedness = ppp
        # Initialize RNGs early (centralized)
        set_seed(seed)
        self.np_random = np_rng()
        self.participation_rng = np_rng_participation()
        self.voting_rng = np_rng_voting()
        self.random = py_rng()
        # Dedicated streams for visualization/debug to avoid perturbing simulation RNG.
        self.rng_viz = np_rng_viz()
        self.rng_debug = np_rng_debug()
        self.random_viz = py_rng_viz()
        self.random_debug = py_rng_debug()
        # Step control
        self.max_steps: Optional[int] = max_steps
        self.running: bool = True
        self.colors = np.arange(num_colors)
        # Cached area-coverage info for fast/exact global distribution updates when areas are disjoint.
        # Initialized to safe defaults because update_global_color_distribution() is called before areas exist.
        self._areas_are_disjoint: bool = False
        self._uncovered_color_counts: np.ndarray = np.zeros(num_colors, dtype=np.int64)
        # Create a scheduler that goes through areas first then color cells
        self.scheduler = CustomScheduler(self)
        # The grid
        # SingleGrid enforces at most one agent per cell;
        # MultiGrid allows multiple agents to be in the same cell.
        self.grid = mesa.space.SingleGrid(height=height, width=width, torus=True)
        # Random bias factors that affect the initial color distribution
        self._vertical_bias = self.random.uniform(0, 1)
        self._horizontal_bias = self.random.uniform(0, 1)
        self._configure_environment_scalars(
            heterogeneity=heterogeneity,
            mu=mu,
            initial_agent_assets=initial_agent_assets,
            election_impact_on_mutation=election_impact_on_mutation,
            color_patches_steps=color_patches_steps,
            patch_power=patch_power,
        )
        self._configure_rules_rewards_and_distance(
            rule_idx=rule_idx,
            distance_idx=distance_idx,
            election_cost_rate=election_cost_rate,
            reward_rate_common=reward_rate_common,
            reward_rate_personal=reward_rate_personal,
            break_even_distance_common=break_even_distance_common,
            break_even_distance_personal=break_even_distance_personal,
            abstention_share=abstention_share,
            num_colors=num_colors,
        )
        # Create search pairs once for faster iterations when comparing orderings
        # (Removed unused self.search_pairs to avoid O(options^2) memory growth.)
        self.option_vec = np.arange(self.options.shape[0])  # Also to speed up
        self.color_search_pairs = list(combinations(range(0, num_colors), 2))
        # Create color cells (IDs start after areas+agents)
        self.color_cells: List[Optional[ColorCell]] = [None] * (height * width)
        self._initialize_color_cells(id_start=num_agents + num_areas)
        # Create voting agents (IDs start after areas)
        self.voting_agents: List[Optional[VoteAgent]] = [None] * num_agents
        self.personality_groups = self.create_personality_groups(num_personality_groups)
        pg_dst = ParticipationModel.pers_dist(num_personality_groups, rng=self.np_random)
        self.initialize_voting_agents(intended_dst=pg_dst, id_start=num_areas)
        self.personality_group_distribution = self._initialize_personality_group_distribution()  # Static
        # Area variables
        self.global_area = self.initialize_global_area()
        self.areas: List[Optional[Area]] = [None] * num_areas
        self._no_overlap = False  # True if areas are instantiated without overlap (speeds up things)
        # Adjust the color pattern to make it less random (see color patches)
        self.adjust_color_pattern(self.color_patches_steps, self.patch_power)
        # Ensure global_color_dst matches the realized grid (not just the preset distribution).
        # This makes step-0 / initialization logs consistent with the actual grid state.
        self.update_global_color_distribution()
        # Create areas
        self.initialize_all_areas()
        # Analyze area coverage once so global distributions can be updated fast + correctly
        # for disjoint area layouts (including layouts with gaps).
        self._analyze_area_coverage()
        # Data collector
        self.datacollector = self.initialize_datacollector()
        # Collect initial data
        self.datacollector.collect(self)

    def _analyze_area_coverage(self) -> None:
        """Compute and cache area coverage/overlap information.

        This is used to safely accelerate global color distribution updates when areas
        are disjoint. If areas overlap, we fall back to grid counting (exact, but slower).
        """
        n_cells = len(self.color_cells)
        membership = np.zeros(n_cells, dtype=np.int16)
        for area in self.areas:
            if area.unique_id == -1:
                continue
            for cell in area.cells:
                idx = self._cell_index_by_pos.get((cell.col, cell.row))
                if idx is not None:
                    membership[idx] += 1
        self._covered_cell_count = int(np.count_nonzero(membership))
        self._areas_are_disjoint = bool(np.max(membership) <= 1)
        # no_overlap means "areas do not overlap" (disjointness only).
        # Full-coverage/partition is tracked separately where needed.
        self._no_overlap = bool(self._areas_are_disjoint)
        # Cache uncovered color counts (uncovered cells are never mutated anywhere).
        uncovered_counts = np.zeros(self.num_colors, dtype=np.int64)
        if self._covered_cell_count < n_cells:
            for i, cell in enumerate(self.color_cells):
                if membership[i] == 0:
                    # If needed in the future, we could save uncovered cells here.
                    uncovered_counts[int(cell.color)] += 1
        self._uncovered_color_counts = uncovered_counts

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

    @property
    def no_overlap(self) -> bool:
        return self._no_overlap

    def _initialize_color_cells(self, id_start=0) -> None:
        """
        Initialize one ColorCell per grid cell.
        Args:
            id_start (int): The starting ID to ensure unique IDs.
        """
        # Map from (col,row) to index in self.color_cells for fast coverage checks.
        self._cell_index_by_pos: dict[tuple[int, int], int] = {}
        # Create a color cell for each cell in the grid
        for idx, (_, (col, row)) in enumerate(self.grid.coord_iter()):
            # Assign unique ID after areas and agents
            unique_id = id_start + idx
            # The colors are chosen by a predefined color distribution
            color = self.color_by_dst_rng(self._preset_color_dst)
            # Create the cell (skip ids for area and voting agents)
            cell = ColorCell(unique_id, self, (col, row), color)
            # Add to the 'model.color_cells' list (for faster access)
            self.color_cells[idx] = cell
            self._cell_index_by_pos[(col, row)] = idx

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
        assets = self.initial_agent_assets
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
        reserved = {(int(x), int(y)) for x in x_coords for y in y_coords}
        # Add additional areas if necessary (num_areas not a square number)
        additional_x, additional_y = [], []
        missing = self.num_areas - len(x_coords) * len(y_coords)
        for _ in range(missing):
            # Avoid placing the "additional" area exactly on the regular grid anchors;
            # otherwise tests/diagnostics can't distinguish them, and we may duplicate placements.
            for _attempt in range(1000):
                rx = int(self.random.randrange(self.grid.width))
                ry = int(self.random.randrange(self.grid.height))
                if (rx, ry) not in reserved:
                    reserved.add((rx, ry))
                    additional_x.append(rx)
                    additional_y.append(ry)
                    break
            else:
                raise RuntimeError("Failed to place all areas. Grid may be too small or num_areas too large.")
                # Fallback: accept any random coordinate (extremely unlikely).
                #additional_x.append(int(self.random.randrange(self.grid.width)))
                #additional_y.append(int(self.random.randrange(self.grid.height)))
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
        def mean_dissatisfaction(m: "ParticipationModel") -> float:
            agents = m.voting_agents
            if not agents:
                return 0.0
            vals = [float(a.dissatisfaction_value) for a in agents if a is not None]
            return float(np.mean(vals)) if vals else 0.0

        return mesa.DataCollector(
            model_reporters={
                "collective_assets": compute_collective_assets,
                "gini_index": compute_gini_index,
                "turnout": get_voter_turnout,
                "mean_p_participation": mean_p_participation,
                "mean_altruism": mean_altruism,
                "mean_dissatisfaction": mean_dissatisfaction,
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
            # Model intentionally stops after the last election-time snapshot;
            # no final "apply pending mutation" step is executed.
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
        self._av_area_color_dst = sums / self.num_areas
        return self._av_area_color_dst


    def update_global_color_distribution(self) -> None:
        """
        This method updates the global color distribution based on the current
        state of the grid. It calculates the distribution of colors across all
        color cells and normalizes it to sum to 1.
        """
        if self._areas_are_disjoint:
            # Fast + exact when areas are disjoint:
            # global counts = sum(area counts) + uncovered counts (uncovered are static).
            counts = np.array(self._uncovered_color_counts, copy=True)
            for area in self.areas:
                if area.unique_id != -1:
                    counts += area.color_counts
            total_cells = len(self.color_cells)
            if total_cells > 0:
                self.global_color_dst = counts / float(total_cells)
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

    @staticmethod
    def _get_voting_rule_conf(rule_idx):
        # Wrap voting rules so they use deterministic RNG
        # Keep self.voting_rule as the base function for tests.
        if rule_idx < 0 or rule_idx >= len(social_welfare_functions):
            raise ValueError(f"rule_idx out of range: {rule_idx} (valid: 0..{len(social_welfare_functions)-1})")
        vr = social_welfare_functions[rule_idx]
        impl_names = [f.__name__ for f in social_welfare_functions]

        # Display names (for UI): prefer short names if aligned; otherwise fallback.
        if len(social_welfare_function_short_names) == len(social_welfare_functions):
            display_names = social_welfare_function_short_names
            display_name = social_welfare_function_short_names[rule_idx]
        else:
            display_names = impl_names
            display_name = str(vr.__name__)

        impl_name = str(vr.__name__)
        return vr, display_names, display_name, impl_names, impl_name

    @staticmethod
    def _get_dist_conf(distance_idx: int):
        """
        Return (callable, display_names, display_name, impl_names, impl_name) for distance_idx.
        Selects an ordering distance (for valid ColorOrderings (permutations)) used in:
          ballot scoring (ScoreVector entries are distances to options)
          rewards (dist_to_reality, personal distance)
        """
        if distance_idx < 0 or distance_idx >= len(distance_functions):
            raise ValueError(
                f"distance_idx out of range: {distance_idx} (valid: 0..{len(distance_functions)-1})"
            )
        f = distance_functions[distance_idx]
        impl_names = [fn.__name__ for fn in distance_functions]
        impl_name = str(f.__name__)

        if len(distance_function_short_names) == len(distance_functions):
            display_names = distance_function_short_names
            display_name = distance_function_short_names[distance_idx]
        else:
            display_names = impl_names
            display_name = impl_name

        return f, display_names, display_name, impl_names, impl_name



def get_color_distribution_function(color: int) -> Callable[[ParticipationModel], float]:
    """
    Returns a lambda to extract a single color's distribution from the model.

    Args:
        color (int): Index of the color.

    Returns:
        Callable[[ParticipationModel], float]: Extractor.
    """
    return lambda m: float(m.global_color_dst[color])
