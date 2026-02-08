from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, cast, List, Optional
from mesa import Agent
# New (minimal) strategy split: participation and voting
from src.agents.strategies import DefaultParticipationStrategy, DefaultVotingStrategy


def _sigmoid(x: float) -> float:
    # Numerically stable-ish sigmoid.
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


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

    def __init__(
        self,
        unique_id,
        model: ParticipationModel,
        pos,
        personality_group,
        personality_group_idx=None,
        assets=1.0,
        add=True,
        participation_strategy=None,
        voting_strategy=None,
    ):
        """ Create a new agent.

        Attributes:
            unique_id: The unique identifier of the agent.
            model: The simulation model of which the agent is part of.
            pos (int, int): The position of the agent in the grid (col, row).
            personality_group: Represents the agent's preferences among colors.
            personality_group_idx: Index of personality group in model's personality_groups list.
            assets: The wealth/assets/motivation of the agent.
            add: Whether to add the agent to the model's agent list and cell.
        """
        super().__init__(unique_id=unique_id, model=model)
        # The "pos" variable in mesa is special, so I avoid it here
        try:
            col, row = pos  # Mesa uses (col, row)
        except ValueError:
            raise ValueError("Position must be a tuple of two integers.")
        self._position = col, row  # Store as (col, row) like mesa standard
        self._assets = float(assets)
        self._num_elections_participated = 0
        self.cell = model.grid.get_cell_list_contents([(col, row)])[0]

        # --- Representation contract (thesis):
        # personality_group: ColorOrdering (permutation)
        # personality: ColorDistribution (per-agent color intensity dist)
        self.personality_group = np.asarray(personality_group)  # ordering / group identity
        self.personality_group_idx = personality_group_idx
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
        self._participating = False
        # Per-election signals (computed right before applying to assets)
        self._delta_abs = 0.0
        self._delta_rel = 0.0

        self.est_real_dist = np.zeros(self.model.num_colors)
        self.confidence = 0.0
        self.award_history: List[float] = []

        # Per-agent personality color-distribution (static), consistent with personality_group.
        self.personal_opt_dist: np.ndarray = self._init_personal_opt_dist()
        # Preferred naming: expose distribution as `.personality`

        # --- Adaptive participation learning (global per agent) ---
        init_q = model.participation_init_q
        self.q_participation = float(init_q)
        self.participation_strategy = (
            participation_strategy if participation_strategy is not None else DefaultParticipationStrategy()
        )
        self.voting_strategy = voting_strategy if voting_strategy is not None else DefaultVotingStrategy()

        # --- Adaptive altruism (reality-weight) learning (per agent) ---
        init_a = model.altruism_init
        self.altruism_factor = float(init_a)

    def __str__(self):
        return (f"Agent(id={self.unique_id}, pos={self.position}, "
                f"pers_group_idx={self.personality_group_idx}, "
                f"personality={self.personality}, assets={self.assets})")

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
    def assets(self) -> float:
        """Return the assets of this agent."""
        return self._assets

    @assets.setter
    def assets(self, value):
        self._assets = float(value)

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
    def personality(self) -> np.ndarray:
        """Per-agent preferred color distribution (ColorDistribution).
        Note: the ordering / group identity is `personality_group`.
        """
        return self.personal_opt_dist

    @property
    def election_delta_abs(self) -> float:
        """Absolute per-election asset delta (pers + common - fee)."""
        return self._reward_pers_comp + self._reward_common_comp - self._fee

    @property
    def election_delta_rel(self) -> float:
        """Relative per-election delta stored at application time.

        Defined as: delta_abs / max(assets_pre, eps) with eps=1.0.
        """
        return float(self._delta_rel)

    @property
    def eligible_for_election(self) -> bool:
        """Whether the agent is eligible for the current election."""
        return self._eligible_for_election

    @property
    def participating(self) -> bool:
        """Whether the agent is participating in the current election (per-election flag)."""
        return bool(self._participating)

    def mark_ineligible_for_election(self) -> None:
        self._eligible_for_election = False

    def set_election_fee(self, fee: float) -> None:
        self._fee = float(fee)

    def add_common_reward(self, amount: float) -> None:
        self._reward_common_comp += float(amount)

    def add_personal_reward(self, amount: float) -> None:
        self._reward_pers_comp += float(amount)

    def reset_reward_variables(self) -> None:
        """Reset per-election variables before the next election."""
        self._eligible_for_election = True
        self._fee = 0.0
        self._reward_pers_comp = 0.0
        self._reward_common_comp = 0.0
        self._participating = False
        self._delta_abs = 0.0
        self._delta_rel = 0.0

    def mark_participating(self) -> None:
        self._participating = True

    def update_known_cells(self, area: Area) -> None:
        """
        This method is to update the list of known cells before casting a vote.
        It is called only by the area during the election process.

        Args:
            area (Area): The area that holds the pool of cells in question
        """
        n_cells = len(area.cells)
        k = len(self.known_cells)
        if n_cells <= 0 or k <= 0:
            self.known_cells = []
            return
        # Sample indices, then index into the list
        if n_cells >= k:
            idx = self.model.np_random.choice(n_cells, size=k, replace=False)
            self.known_cells = [area.cells[int(i)] for i in idx]
        else:
            self.known_cells = list(area.cells)

    def reward_agent(self) -> None:
        """Reward the agent by increasing/decreasing her assets.

        Computes and stores per-election signals *before* mutating assets:
          - delta_abs: pers + common - fee
          - delta_rel: delta_abs / max(assets_pre, 1.0)

        And saves delta_abs into award_history.
        """
        assets_pre = float(self.assets)
        delta_abs = float(self.election_delta_abs)
        self._delta_rel = float(delta_abs / max(assets_pre, 1.0))

        self.award_history.append(delta_abs)
        self.assets += delta_abs
        if self.assets < 0:
            self.assets = 0  # Ensure assets don't go negative

    def ask_for_participation(self, area: Area) -> bool:
        """
        Decide whether to participate in the given area's election.
        """
        return self.participation_strategy.decide_participation(self, area)

    def vote(self, area: Area):
        """Return raw oppose-scores (ScoreVector) over all options.

        Contract:
        - shape = (num_options,)
        - values in [0,1]
        - lower = better
        - NOT normalized

        Sampling of known_cells happens only in Area._tally_votes().
        """
        if TYPE_CHECKING:
            self.model = cast(ParticipationModel, self.model)
        options = self.model.options
        return self.voting_strategy.score_options(self, area, options)

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
        beta = self.model.participation_beta
        q = self.q_participation
        return _sigmoid(beta * q)

    def apply_participation_update(self, delta_assets: float) -> None:
        """Naive action reinforcement update for q_participation.

        Contract (thesis baseline): reinforce last action.
        - participating + positive delta => q up (p up)
        - abstained     + positive delta => q down (p down)
        - participating + negative delta => q down
        - abstained     + negative delta => q up
        """
        alpha = self.model.participation_alpha
        sign = 1.0 if self._participating else -1.0
        q = self.q_participation + alpha * sign * delta_assets
        q_max = self.model.participation_q_max
        if q_max > 0:
            q = float(np.clip(q, -q_max, q_max))
        self.q_participation = q

    def apply_altruism_update(self, delta_assets: float) -> None:
        """Participant-only learning of altruism_factor (reality-weight).
        Update rule:
            a = a + altruism_alpha * delta_assets
            a = clip(a, [altruism_clip_min, altruism_clip_max])
        """
        if not self.participating:
            return
        alpha = float(self.model.altruism_alpha)
        if alpha == 0.0:
            return

        a = float(self.altruism_factor)
        a = a + alpha * float(delta_assets)

        lo = float(self.model.altruism_clip_min)
        hi = float(self.model.altruism_clip_max)
        a = float(np.clip(a, lo, hi))
        self.altruism_factor = a

    def _init_personal_opt_dist(self) -> np.ndarray:
        """Create a per-agent personal_opt_dist (distribution)
        consistent with the agent's personality_group.

        Contract:
        - nonnegative
        - sums to 1
        - argsort(personal_opt_dist)[::-1] equals personality_group
        """
        # Fallback if no proper model personality_group context exists (DummyModel).
        num_colors = int(self.model.num_colors)
        if num_colors <= 0:
            return np.asarray([], dtype=np.float32)

        personality_group = np.asarray(self.personality_group)
        conc = self.model.personal_opt_dist_concentration
        conc = max(conc, 1e-8)  # Avoid zero concentration

        # Sample positive intensities, sort descending, then assign by rank position.
        rng = self.model.np_random
        vals = rng.exponential(scale=1.0, size=num_colors).astype(np.float64)
        # Concentration: >1 makes the distribution more peaked; <1 flattens.
        vals = np.power(vals + 1e-12, conc)
        vals.sort()
        vals = vals[::-1]

        # Assign values according to personality_group ordering.
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
