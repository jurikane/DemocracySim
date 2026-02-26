from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, cast, List, Optional
from mesa import Agent
# New (minimal) strategy split: participation and voting
from src.agents.strategies import DefaultParticipationStrategy, DefaultVotingStrategy
from src.utils.distance_functions import distribution_distance_l1


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
        # Dissatisfaction value placeholder (learning signal for altruism).
        # NOTE: computed each step; initialized to 0.0 until first update.
        self.dissatisfaction_value: float = 0.0
        # EMA baseline placeholder (computed each step alongside dissatisfaction).
        # Initialized as NaN until first update.
        self.dissatisfaction_baseline: float = float("nan")
        # Dissatisfaction signal for learning (dv - baseline).
        self.dissatisfaction_signal: float = 0.0
        self._num_elections_participated = 0
        self.cell = model.grid.get_cell_list_contents([(col, row)])[0]

        # --- Representation contract (thesis):
        # personality_group: ColorOrdering (permutation)
        # personality: ColorDistribution (per-agent color intensity dist)
        self.personality_group = np.asarray(personality_group)  # ordering / group identity
        self.personality_group_idx = personality_group_idx
        # ColorCell objects the agent knows (knowledge)
        self.known_cells: List[Optional[ColorCell] | int] = [None] * model.known_cells
        if add:  # Add the agent to the models' agent list and the cell
            model.voting_agents.append(self)
            cell = model.grid.get_cell_list_contents([(col, row)])[0]
            cell.add_agent(self)
        # Election relevant variables
        self._eligible_for_election = True
        self._fee = 0.0
        self._reward_personal = 0.0
        self._participating = False
        # Per-election signals (computed right before applying to assets)
        self._delta_abs = 0.0
        self._delta_rel = 0.0

        self.est_real_dist = np.zeros(self.model.num_colors)
        self.confidence = 0.0
        self.award_history: List[float] = []

        # Per-agent personality color-distribution (static), consistent with personality_group.
        self.personal_opt_dist: np.ndarray = self._init_personal_opt_dist()
        # Precomputed self-regarding option-oppose scores (static per agent/model setup).
        self.self_regarding_oppose_scores: np.ndarray = self._compute_self_regarding_oppose_scores()
        # Per-vote mode marker used for diagnostics/logging.
        # True=altruistic, False=self-regarding, None=unknown.
        self.voted_altruistically: bool | None = None
        # Preferred naming: expose distribution as `.personality`

        # --- Adaptive participation learning (global per agent) ---
        init_q = model.participation_init_q
        self.q_participation = float(init_q)
        # EMA baseline and signal for participation learning.
        self.participation_baseline: float = float("nan")
        self.participation_signal: float = 0.0
        self.participation_signal_group_component: float = 0.0
        self.participation_signal_fee_component: float = 0.0
        self.participation_strategy = (
            participation_strategy if participation_strategy is not None else DefaultParticipationStrategy()
        )
        self.voting_strategy = voting_strategy if voting_strategy is not None else DefaultVotingStrategy()

        # --- Altruism behavior state (vote-mode probability) ---
        if str(getattr(model, "altruism_mode", "static")) == "static":
            self.altruism_factor = float(model.altruism_static)
        else:
            self.altruism_factor = float(model.altruism_init)

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
        """Realized absolute per-election asset delta after asset-floor clamp."""
        return float(self._delta_abs)

    @property
    def election_delta_rel(self) -> float:
        """Relative per-election delta stored at application time.

        Defined as: realized_delta_abs / assets_pre for assets_pre > 0, else 0.0.
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

    @property
    def election_fee(self) -> float:
        """Election participation fee component for current step."""
        return float(self._fee)

    @property
    def reward_personal(self) -> float:
        """Per-election reward amount for current step."""
        return float(self._reward_personal)

    def mark_ineligible_for_election(self) -> None:
        self._eligible_for_election = False

    def set_election_fee(self, fee: float) -> None:
        self._fee = float(fee)

    def add_personal_reward(self, amount: float) -> None:
        self._reward_personal += float(amount)

    def reset_reward_variables(self) -> None:
        """Reset per-election variables before the next election."""
        self._eligible_for_election = True
        self._fee = 0.0
        self._reward_personal = 0.0
        self._participating = False
        self._delta_abs = 0.0
        self._delta_rel = 0.0
        self.voted_altruistically = None

    def mark_participating(self) -> None:
        self._participating = True

    def update_known_cells(self, area: Area) -> None:
        """
        This method is to update the list of known cells before casting a vote.
        It is called only by the area during the election process.

        Args:
            area (Area): The area that holds the pool of cells in question
        """
        quality_target_mode = str(getattr(self.model, "quality_target_mode", "reality"))
        if quality_target_mode == "puzzle":
            k = int(getattr(self.model, "known_cells", len(self.known_cells)))
            c = int(self.model.num_colors)
            if c <= 0 or k <= 0:
                self.known_cells = []
                return

            puzzle = getattr(area, "puzzle_distribution", None)
            probs = np.asarray(puzzle, dtype=np.float64) if puzzle is not None else np.asarray([], dtype=np.float64)
            if probs.ndim != 1 or probs.size != c or not np.all(np.isfinite(probs)):
                probs = np.full(c, 1.0 / float(c), dtype=np.float64)
            probs = np.clip(probs, 0.0, np.inf)
            total = float(np.sum(probs))
            if total <= 0.0:
                probs = np.full(c, 1.0 / float(c), dtype=np.float64)
            else:
                probs = probs / total
            draws = self.model.rng_puzzle.choice(c, size=k, replace=True, p=probs)
            self.known_cells = [int(x) for x in np.asarray(draws, dtype=np.int16).tolist()]
            return

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

        Computes and stores realized per-election signals:
          - raw_delta_abs: pers + common - fee
          - assets_post = max(0, assets_pre + raw_delta_abs)
          - delta_abs: assets_post - assets_pre
          - delta_rel: delta_abs / assets_pre (if assets_pre > 0 else 0)

        And saves delta_abs into award_history.
        """
        assets_pre = float(self.assets)
        raw_delta_abs = float(self._reward_personal - self._fee)
        assets_post = max(0.0, assets_pre + raw_delta_abs)
        self._delta_abs = float(assets_post - assets_pre)
        self._delta_rel = float(self._delta_abs / assets_pre) if assets_pre > 0.0 else 0.0

        self.award_history.append(self._delta_abs)
        self.assets = assets_post

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

        Sampling of known_cells happens in Area.step().
        """
        if TYPE_CHECKING:
            self.model = cast(ParticipationModel, self.model)
        options = self.model.options
        scores = self.voting_strategy.score_options(self, area, options)
        if not isinstance(getattr(self, "voted_altruistically", None), bool):
            self.voted_altruistically = None
        return scores

    def _knowledge_distribution(self, area: Area) -> np.ndarray:
        """Return the agent's knowledge-based distribution."""
        if not self.known_cells:
            raise ValueError("Agent has no known cells to estimate distribution.")
        dist, _ = self.estimate_real_distribution(area)
        return np.asarray(dist, dtype=np.float64)

    def compute_dissatisfaction_value(self, *, area: Area, model: ParticipationModel) -> float:
        """Compute dissatisfaction value (distance) between personality and a target distribution."""
        personality = np.asarray(self.personality, dtype=np.float64)
        area_dist = np.asarray(area.color_distribution, dtype=np.float64)
        global_color_dst = np.asarray(model.global_color_dst, dtype=np.float64)

        mode = model.satisfaction_mode
        if mode == "global":
            target = global_color_dst
            sv = distribution_distance_l1(personality, target)
        elif mode == "area":
            sv = distribution_distance_l1(personality, area_dist)
        elif mode == "knowledge":
            target = self._knowledge_distribution(area)
            sv = distribution_distance_l1(personality, target)
        elif mode == "combination":
            d_global = distribution_distance_l1(personality, global_color_dst)
            d_area = distribution_distance_l1(personality, area_dist)
            d_knowledge = distribution_distance_l1(personality, self._knowledge_distribution(area))
            sv = (d_global + d_area + d_knowledge) / 3.0
        else:
            raise ValueError(f"Unsupported satisfaction_mode: {mode}")
        return float(sv)

    def estimate_real_distribution(self, area: Area) -> tuple[np.ndarray, float]:
        """
        The agent estimates the real color distribution in the area based on
        her own knowledge (self.known_cells).

        Args:
            area (Area): The area the agent uses to estimate.

        Returns:
            tuple[np.array, float]: (distribution, confidence)
        """
        known_colors_list: list[int] = []
        for c in (self.known_cells or []):
            if c is None:
                continue
            if isinstance(c, (int, np.integer)):
                known_colors_list.append(int(c))
            else:
                known_colors_list.append(int(c.color))
        known_colors = np.asarray(known_colors_list, dtype=np.int16)
        if known_colors.size == 0:
            # No information -> uniform estimate, zero confidence.
            c = int(self.model.num_colors)
            if c > 0:
                self.est_real_dist[:] = 1.0 / c
            self.confidence = 0.0
            return self.est_real_dist, self.confidence
        # Get the unique color ids present and count their occurrence
        unique, counts = np.unique(known_colors, return_counts=True)
        # Update the est_real_dist and confidence values of the agent
        self.est_real_dist.fill(0)  # To ensure the ones not in unique are 0
        self.est_real_dist[unique] = counts / float(known_colors.size)
        denom = float(area.num_cells) if getattr(area, "num_cells", 0) else 0.0
        self.confidence = float(known_colors.size / denom) if denom > 0 else 0.0
        return self.est_real_dist, self.confidence

    def participation_probability(self) -> float:
        """Current learned participation probability p in [0,1]."""
        beta = self.model.participation_beta
        q = self.q_participation
        return _sigmoid(beta * q)

    def apply_participation_update(self, participation_signal: float) -> None:
        """Naive action reinforcement update for q_participation.

        Contract (thesis baseline): reinforce last action.
        - participating + positive signal => q up (p up)
        - abstained     + positive signal => q down (p down)
        - participating + negative signal => q down
        - abstained     + negative signal => q up

        Note: participation_signal is the realized level signal (delta_rel).
        The EMA baseline is tracked for diagnostics only.
        """
        alpha = self.model.participation_alpha
        sign = 1.0 if self._participating else -1.0
        q = self.q_participation + alpha * sign * participation_signal
        q_max = self.model.participation_q_max
        if q_max > 0:
            q = float(np.clip(q, -q_max, q_max))
        self.q_participation = q

    def apply_altruism_update(self, dissatisfaction_signal: float) -> None:
        """Participant-only learning of altruism_factor (reality-weight).

        Update rule:
            a = a - altruism_alpha * dissatisfaction_signal
            a = clip(a, [altruism_clip_min, altruism_clip_max])
        """
        if not self.participating:
            return
        alpha = float(self.model.altruism_alpha)
        if alpha == 0.0:
            return
        if not np.isfinite(dissatisfaction_signal):
            return

        a = float(self.altruism_factor)
        a = a - alpha * float(dissatisfaction_signal)

        lo = float(self.model.altruism_clip_min)
        hi = float(self.model.altruism_clip_max)
        a = float(np.clip(a, lo, hi))
        self.altruism_factor = a

    def apply_altruism_satisfaction_mode(self, dissatisfaction_value: float) -> None:
        """Set/update altruism from current dissatisfaction level (pre-election).

        Satisfaction-mode contract:
        - dissatisfaction_value is a normalized distance in [0,1]
        - s = 1 - dissatisfaction_value
        - target altruism = sigmoid(k * (s - theta))
        - gamma==1 => direct mapping (a := target)
        - gamma<1  => EMA-like smoothing toward target
        """
        if not np.isfinite(dissatisfaction_value):
            return
        d = float(np.clip(float(dissatisfaction_value), 0.0, 1.0))
        s = 1.0 - d
        theta = float(getattr(self.model, "altruism_satisfaction_theta", 0.5))
        theta = float(np.clip(theta, 0.0, 1.0))
        slope = float(getattr(self.model, "altruism_satisfaction_slope", 4.0))
        if not np.isfinite(slope) or slope <= 0.0:
            return
        z = float(slope * (s - theta))
        target = float(1.0 / (1.0 + np.exp(-z)))
        gamma = float(getattr(self.model, "altruism_response_gamma", 1.0))
        gamma = float(np.clip(gamma, 0.0, 1.0))
        if gamma >= 1.0:
            a = target
        else:
            a_prev = float(self.altruism_factor)
            a = (1.0 - gamma) * a_prev + gamma * target
        lo = float(self.model.altruism_clip_min)
        hi = float(self.model.altruism_clip_max)
        self.altruism_factor = float(np.clip(a, lo, hi))

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
        peakedness = float(self.model.personal_preference_peakedness)

        # Sample positive intensities, sort descending, then assign by rank position.
        rng = self.model.np_random
        vals = rng.exponential(scale=1.0, size=num_colors).astype(np.float64)
        # Concentration: >1 makes the distribution more peaked; <1 flattens.
        vals = np.power(vals + 1e-12, peakedness)
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

    def _compute_self_regarding_oppose_scores(self) -> np.ndarray:
        """Compute static self-regarding oppose scores against all options.

        Self-regarding mode uses personality_group ordering directly (no knowledge estimate).
        """
        options = np.asarray(self.model.options)
        target_ordering = np.asarray(self.personality_group, dtype=np.int64)
        dist_func = self.model.distance_func
        search_pairs = self.model.color_search_pairs
        scores = np.zeros(int(options.shape[0]), dtype=np.float32)
        for i, opt in enumerate(options):
            scores[i] = np.float32(
                float(dist_func(target_ordering, np.asarray(opt, dtype=np.int64), search_pairs))
            )
        return scores
