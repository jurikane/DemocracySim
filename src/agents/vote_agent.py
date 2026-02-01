from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, cast, List, Optional, Protocol, Any
from mesa import Agent


def _sigmoid(x: float) -> float:
    # Numerically stable-ish sigmoid.
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


class Policy(Protocol):
    def decide_participation(self, agent, area) -> bool: ...
    def decide_altruism_factor(self, agent, area) -> float: ...
    def rank_options(self, agent, area, options: Any) -> np.ndarray: ...


class ParticipationPolicy:
    """Default policy: adaptive probabilistic participation + random altruism + distance-based ranking."""

    def decide_participation(self, agent, area) -> bool:
        # Global, non-strategic participation policy with explicit learning state:
        # p = sigmoid(beta * q_participation)
        beta = float(getattr(agent.model, "participation_beta"))
        q = float(getattr(agent, "q_participation"))
        p = _sigmoid(beta * q)
        # Use the model-level seeded NumPy RNG for determinism.
        return bool(float(agent.model.np_random.random()) < p)

    def decide_altruism_factor(self, agent, area) -> float:
        # TODO do this properly
        # should we change the name "altruism_factor" to "cooperation_factor" or "reality-weight"?
        return agent.random.uniform(0.0, 1.0)

    def rank_options(self, agent, area, options: Any) -> np.ndarray:
        # Use existing distance function; personality_group is an ORDERING.
        dist_func = agent.model.distance_func
        ranking = np.zeros(options.shape[0])
        color_search_pairs = agent.model.color_search_pairs
        for i, option in enumerate(options):
            ranking[i] = dist_func(agent.personality_group, option, color_search_pairs)
        ranking /= ranking.sum() if ranking.sum() else 1.0
        return ranking


if TYPE_CHECKING:  # Type hint for IDEs
    from src.models.participation_model import ParticipationModel
    from src.agents.color_cell import ColorCell
    from src.agents.area import Area


def combine_and_normalize(arr_1: np.ndarray, arr_2: np.ndarray, factor: float):
    """
    Combine two arrays weighted by a factor favoring arr_1.
    The first array is to be the estimated real distribution.
    And the other is to be the personality_group vector of the agent.

    Args:
        arr_1 (np.array): Estimated real distribution.
        arr_2 (np.array): Personality group vector.
        factor (float): Weight for arr_1.

    Returns:
        result (np.array): Normalized weighted linear combination.

    Example:
        TODO
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
    """An agent with resources and preferences that may participate in elections."""

    def __init__(self, unique_id, model: ParticipationModel, pos,
                 personality_group, personality_group_idx=None, assets=1, add=True, policy: Policy | None = None):
        """ Create a new agent.

        Attributes:
            unique_id: The unique identifier of the agent.
            model: The simulation model of which the agent is part of.
            pos (int, int): The position of the agent in the grid (col, row).
            personality_group: Represents the agent's preferences among colors.
            personality_group_idx: Index of personality group in model's personality_groups list.
            assets: The wealth/assets/motivation of the agent.
            add: Whether to add the agent to the model's agent list and cell.
            policy: The behavior strategy of the agent.
        """
        super().__init__(unique_id=unique_id, model=model)
        # The "pos" variable in mesa is special, so I avoid it here
        try:
            col, row = pos  # Mesa uses (col, row)
        except ValueError:
            raise ValueError("Position must be a tuple of two integers.")
        self._position = col, row  # Store as (col, row) like mesa standard
        self._assets = assets
        self._num_elections_participated = 0

        # --- Representation contract (thesis):
        # personality_group: ColorOrdering (permutation)
        # personality: ColorDistribution (per-agent color intensity dist)
        self.personality_group = np.asarray(personality_group)  # ordering / group identity
        self.personality_group_idx = personality_group_idx

        # Backward-compat: some code/tests still expect ordering under `.personality`.
        # We keep an explicit accessor for that ordering.
        # (Do NOT use `.personality_group_ordering` for new code; use `.personality_group`.)

        self.cell = model.grid.get_cell_list_contents([(col, row)])[0]
        # ColorCell objects the agent knows (knowledge)
        self.known_cells: List[Optional[ColorCell]] = [None] * model.known_cells
        if add:  # Add the agent to the models' agent list and the cell
            model.voting_agents.append(self)
            cell = model.grid.get_cell_list_contents([(col, row)])[0]
            cell.add_agent(self)
        # Election relevant variables
        self._eligible_for_election = True
        self._fee = 0.0
        self._reward_pers_comp = 0.0
        self._reward_common_comp = 0.0

        self.est_real_dist = np.zeros(self.model.num_colors)
        self.confidence = 0.0
        self.award_history: List[float] = []

        # Per-agent personality color-distribution (static), consistent with personality_group.
        self.personal_opt_dist: np.ndarray = self._init_personal_opt_dist()
        # Preferred naming: expose distribution as `.personality`

        # --- Adaptive participation learning (global per agent) ---
        init_q = getattr(model, "participation_init_q", 0.0)
        self.q_participation = float(init_q)

        # Policy (behavior strategy)
        self.policy: Policy = policy if policy is not None else ParticipationPolicy()

    def __str__(self):
        return (f"Agent(id={self.unique_id}, pos={self.position}, "
                f"personality_group={self.personality_group}, assets={self.assets})")

    @property
    def position(self) -> tuple:
        """Return the location of the agent.
        Logic: (col, row), following Mesa conventions"""
        return self._position

    @property
    def col(self) -> int:
        """Return the col location of the agent."""
        return self._position[0]

    @property
    def row(self) -> int:
        """Return the row location of the agent."""
        return self._position[1]

    @property
    def assets(self) -> int:
        """Return the assets of this agent."""
        return self._assets

    @assets.setter
    def assets(self, value):
        self._assets = value

    @assets.deleter
    def assets(self):
        del self._assets

    @property
    def num_elections_participated(self) -> int:
        """Return the number of elections this agent has participated in."""
        return self._num_elections_participated

    @num_elections_participated.setter
    def num_elections_participated(self, value):
        self._num_elections_participated = value

    @property
    def election_delta_signal(self) -> float:
        """Return the per-election asset delta signal for participation learning."""
        return self._reward_pers_comp + self._reward_common_comp - self._fee

    @property
    def eligible_for_election(self) -> bool:
        return self._eligible_for_election

    def mark_ineligible_for_election(self) -> None:
        self._eligible_for_election = False

    def set_election_fee(self, fee: float) -> None:
        self._fee = float(fee)

    def add_common_reward(self, amount: float) -> None:
        self._reward_common_comp += float(amount)

    def add_personal_reward(self, amount: float) -> None:
        self._reward_pers_comp += float(amount)

    def reset_election_variables(self) -> None:
        """Reset per-election variables before the next election."""
        self._eligible_for_election = True
        self._fee = 0.0
        self._reward_pers_comp = 0.0
        self._reward_common_comp = 0.0

    def update_known_cells(self, area: Area) -> None:
        """
        This method is to update the list of known cells before casting a vote.

        Args:
            area (Area): The area that holds the pool of cells in question
        """
        n_cells = len(area.cells)
        k = len(self.known_cells)
        self.known_cells = (
            self.random.sample(area.cells, k)
            if n_cells >= k
            else area.cells
        )

    def reward_agent(self) -> None:
        """
        Reward the agent by increasing/decreasing her assets.
        And save the awarded amount in the agent's history.
        """
        total_asset_delta = self.election_delta_signal
        self.award_history.append(total_asset_delta)
        self.assets += total_asset_delta
        if self.assets < 0:
            self.assets = 0  # Ensure assets don't go negative

    def ask_for_participation(self, area: Area) -> bool:
        """
        Decide whether to participate in the given area's election.

        Args:
            area (Area): The area in which the election takes place.

        Returns:
            True if the agent decides to participate, False otherwise
        """
        #print("Agent", self.unique_id, "decides whether to participate",
        #      "in election of area", area.unique_id)
        # TODO Implement this (is to be decided upon a learned decision tree)
        # Delegate to policy for decision
        return self.policy.decide_participation(self, area)

    def decide_altruism_factor(self, area: Area) -> float:
        """
        Uses a trained decision tree to decide on the altruism factor.

        Returns:
            float
        """
        # TODO Implement this (is to be decided upon a learned decision tree)
        # This part is important - also for monitoring - save/plot a_factors
        a_factor = self.policy.decide_altruism_factor(self, area)
        return a_factor

    def compute_assumed_opt_dist(self, area: Area) -> np.ndarray:
        """Compute the distribution the agent uses as its internal 'ideal' for voting.

        Mix self-interest vs reality-tracking.
        - self-interest is represented by personal_opt_dist (static, per agent)
        - reality-tracking is represented by the agent's estimated reality

        altruism_factor semantics:
        - 0.0 => purely self-interest (personal_opt_dist)
        - 1.0 => purely reality-tracking (est_real_dist)

        Args:
            area (Area): The area the agent is voting in.
        Returns:
            np.ndarray: The assumed optimal color distribution (normalized).
        """
        a_factor = float(self.decide_altruism_factor(area))
        # Clamp for safety (policy may not respect bounds yet)
        a_factor = float(np.clip(a_factor, 0.0, 1.0))

        est_dist, _conf = self.estimate_real_distribution(area)
        personal = np.asarray(self.personal_opt_dist, dtype=np.float32)
        if personal.ndim != 1 or personal.shape[0] != est_dist.shape[0]:
            raise ValueError("personal_opt_dist shape mismatch with estimated distribution")
        # Combine and normalize to a distribution
        return combine_and_normalize(est_dist, personal, a_factor)

    def vote(self, area: Area):
        """Return a normalized 'oppose score' vector over all options.

        Lower score = better (less opposition / closer to the agent's assumed-optimal).
        """
        if TYPE_CHECKING:  # Type hint for IDEs
            self.model = cast(ParticipationModel, self.model)

        options = self.model.options
        # Delegate to policy ranking
        ranking = self.policy.rank_options(self, area, options)
        return ranking

    def estimate_real_distribution(self, area: Area) -> tuple[np.ndarray, float]:
        """
        The agent estimates the real color distribution in the area based on
        her own knowledge (self.known_cells).

        Args:
            area (Area): The area the agent uses to estimate.

        Returns:
            tuple[np.array, float]: (distribution, confidence)
        """
        known_colors = np.array([cell.color for cell in self.known_cells])
        # Get the unique color ids present and count their occurrence
        unique, counts = np.unique(known_colors, return_counts=True)
        # Update the est_real_dist and confidence values of the agent
        self.est_real_dist.fill(0)  # To ensure the ones not in unique are 0
        self.est_real_dist[unique] = counts / known_colors.size
        self.confidence = len(self.known_cells) / area.num_cells
        return self.est_real_dist, self.confidence

    def participation_probability(self) -> float:
        """Current learned participation probability p in [0,1]."""
        beta = float(getattr(self.model, "participation_beta", 1.0))
        return _sigmoid(beta * float(self.q_participation))

    def apply_participation_update(self, delta_assets: float) -> None:
        """Update q_participation from a realized per-election asset delta."""
        alpha = float(getattr(self.model, "participation_alpha", 0.0))
        # learning signal (delta_assets) = per-election change in assets
        q = (1.0 - alpha) * float(self.q_participation) + alpha * float(delta_assets)
        q_max = float(getattr(self.model, "participation_q_max", 0.0))
        if q_max > 0:
            q = float(np.clip(q, -q_max, q_max))
        self.q_participation = float(q)

    @property
    def personality_group_ordering(self) -> np.ndarray:
        """Backward-compat accessor for the ordering (ColorOrdering)."""
        return self.personality_group

    @property
    def personality(self) -> np.ndarray:
        """Per-agent preferred color distribution (ColorDistribution).

        Note: the ordering / group identity is `personality_group`.
        """
        return self.personal_opt_dist

    def _init_personal_opt_dist(self) -> np.ndarray:
        """Create a per-agent personal_opt_dist (distribution)
        consistent with the agent's personality_group.

        Contract:
        - nonnegative
        - sums to 1
        - argsort(personal_opt_dist)[::-1] equals personality_group
        """
        # Fallback if no proper model personality_group context exists (DummyModel).
        num_colors = int(getattr(self.model, "num_colors") or 0)
        if num_colors <= 0:
            return np.asarray([], dtype=np.float32)

        personality_group = np.asarray(self.personality_group)
        conc = getattr(self.model, "personal_opt_dist_concentration", 1.0)
        conc = max(conc, 1e-8)  # Avoid zero concentration

        # Sample positive intensities, sort descending, then assign by rank position.
        rng = self.model.np_random
        vals = rng.exponential(scale=1.0, size=num_colors).astype(np.float64)
        # Concentration: >1 makes the distribution more peaked; <1 flattens.
        vals = np.power(vals + 1e-12, conc)
        vals.sort()
        vals = vals[::-1]

        # Assign values according to personality_group ranking.
        dist = np.zeros(num_colors, dtype=np.float64)
        for rank_pos in range(num_colors):
            color = int(personality_group[rank_pos])
            dist[color] = float(vals[rank_pos])

        # Normalize to sum to 1.
        s = float(dist.sum())
        if s <= 0:
            dist[:] = 1.0 / num_colors
        else:
            dist /= s

        return dist.astype(np.float32)

