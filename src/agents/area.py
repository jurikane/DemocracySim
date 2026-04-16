from __future__ import annotations
from typing import TYPE_CHECKING, cast, List
import numpy as np
from mesa import Agent
if TYPE_CHECKING:  # Type hint for IDEs
    from src.models.participation_model import ParticipationModel
    from src.agents.color_cell import ColorCell
from src.agents.vote_agent import VoteAgent
from src.utils.representations import (
    distribution_to_ordering_tie_aware,
    scores_to_ordering,
    validate_score_vector_unit_interval,
)
# from src.utils.rng import np_rng_debug

# Center-biased symmetric Dirichlet priors for puzzle generation.
# Per-component concentration keeps behavior comparable across num_colors.
_PUZZLE_ALPHA_CENTER_PER_COMPONENT = 2.0
_PUZZLE_ALPHA_SHOCK_PER_COMPONENT = 2.0


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
        super().__init__(unique_id=unique_id, model=model)
        self.np_random = model.random  # Use the model's random generator for reproducibility
        self._set_dimensions(width, height, size_variance)
        self.agents: List["VoteAgent"] = []
        self._personality_group_distribution = None
        self._personality_group_counts: list[int] = []
        self.cells: List["ColorCell"] = []
        self._idx_field = None  # An indexing position of the area in the grid
        self._color_distribution = np.zeros(model.num_colors) # Initialize to 0
        # Canonical integer counts (distribution is derived).
        self._color_counts = np.zeros(model.num_colors, dtype=np.int64)
        self._voted_ordering = None
        self._voter_turnout = 0  # In percent
        self._dist_to_reality = None  # Elected vs. actual color distribution
        self._puzzle_distance = None  # Elected vs. puzzle ordering distance
        self._puzzle_distribution = None  # Optional puzzle distribution for this step
        self._grid_ordering_for_quality = None
        self._puzzle_ordering_for_quality = None
        self._group_outcome_distance: list[float] = []
        self._election_fee_pool: float = 0
        self._num_agents_participated_last = None  # For statistics
        self._num_eligible_voters_last = 0  # Eligible population count for diagnostics/logging
        self._diag_history: List[dict] = []  # Per-area diagnostics time series
        self._debug_history: List[dict] = []  # Per-area debug snapshots (optional)
        self._debug_last_votes: List[dict] = []  # Per-step vote records (optional)

    def __str__(self):
        return (f"Area(id={self.unique_id}, size={self._height}x{self._width}, "
                f"at idx_field={self._idx_field}, "
                f"num_agents={self.num_agents}, num_cells={self.num_cells}, "
                f"color_distribution={self.color_distribution})")

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def num_cells(self):
        return self._width * self._height

    @property
    def personality_group_distribution(self):
        return self._personality_group_distribution

    @property
    def personality_group_counts(self) -> list[int]:
        return self._personality_group_counts

    @property
    def color_distribution(self):
        return self._color_distribution

    @property
    def color_counts(self) -> np.ndarray:
        """Integer counts of colors in this area (canonical state)."""
        return self._color_counts

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
    def puzzle_distance(self):
        return self._puzzle_distance

    @property
    def puzzle_distribution(self):
        return self._puzzle_distribution

    @property
    def puzzle_mode(self) -> bool:
        return self.model.is_puzzle_mode

    @property
    def group_outcome_distance(self) -> list[float]:
        return list(self._group_outcome_distance)

    @property
    def election_fee_pool(self) -> float:
        return float(self._election_fee_pool)

    @property
    def diag_history(self) -> List[dict]:
        return self._diag_history

    @property
    def debug_history(self) -> List[dict]:
        return self._debug_history

    @property
    def idx_field(self):
        return self._idx_field

    @property
    def num_agents_participated_last(self):
        return self._num_agents_participated_last

    @num_agents_participated_last.setter
    def num_agents_participated_last(self, n: int):
        self._num_agents_participated_last = n

    @property
    def num_eligible_voters_last(self) -> int:
        return int(self._num_eligible_voters_last)

    @num_eligible_voters_last.setter
    def num_eligible_voters_last(self, n: int):
        self._num_eligible_voters_last = int(n)

    @idx_field.setter
    def idx_field(self, pos: tuple):
        """
        Sets the indexing field (cell coordinate in the grid) of the area.

        This method sets the areas indexing-field (top-left cell coordinate)
        which determines which cells and agents on the grid belong to the area.
        The cells and agents are added to the area's lists of cells and agents.

        This is write-once. Area membership is built when idx_field is set.
        If agents are added to the area later, they must be registered manually.

        Args:
            pos: (x, y) representing the areas top-left coordinates.
        """
        if self._idx_field is not None:  # Write-once.
            raise RuntimeError("idx_field already set; areas are static")
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
        self.update_color_distribution()
        self._update_personality_group_distribution()

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
            self._width = max(1, int(width * w_var_factor))
            self.width_off = abs(width - self._width)
            self._height = max(1, int(height * h_var_factor))
            self.height_off = abs(height - self._height)

    def _update_personality_group_distribution(self) -> None:
        """
        This method calculates the areas current distribution of personality (preference) groups.
        """
        counts = [0] * self.model.num_personality_groups
        for agent in self.agents:
            counts[agent.personality_group_idx] += 1
        self._personality_group_counts = counts
        if self.num_agents == 0:
            self._personality_group_distribution = [0.0] * self.model.num_personality_groups
        else:
            self._personality_group_distribution = [
                count / self.num_agents
                for count in counts
            ]

    def add_agent(self, agent: VoteAgent) -> None:
        """
        Appends an agent to the areas agents list.

        Args:
            agent (VoteAgent): The agent to be added to the area.
        """
        # Make sure it's an instance of Agent
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

    def conduct_election(self) -> int:
        """
        Simulates the election within the area and manages rewards.

        The election process asks agents to participate, collects votes,
        aggregates preferences using the model's voting rule,
        and saves the elected option as the latest winning option.
        Agents incur costs for participation
        and may receive rewards based on the outcome.

        Returns:
            int: The voter turnout in percent over resident area population.
            Returns 0 if no agent participates.
        """
        # Ask agents for participation and their votes
        preference_profile = self._tally_votes()
        n_eligible = int(sum(1 for a in self.agents if a.eligible_for_election))
        n_resident = int(self.num_agents)
        self.num_eligible_voters_last = n_eligible
        # Check for the case that no agent participated
        if preference_profile.ndim != 2 or preference_profile.shape[0] == 0:
            # Set to previous outcome but don't distribute rewards as usual
            # print("Area", self.unique_id, "no one participated in the election")
            # If no previous outcome, use the real distribution ordering
            real_color_ord = self._ordering_from_distribution_tie_aware(
                self.color_distribution,
                reference_ordering=self._voted_ordering,
                rng=self.model.voting_rng,
            )
            # Assumption is: if no (new) decision is made, things stay the same.
            #   Alternative to think about: randomly select any available option.
            if self._voted_ordering is None:
                self._voted_ordering = real_color_ord
            # Update quality distances for monitoring even when no one participated.
            self._update_quality_distances()
            self._update_group_outcome_distance()
            # Thought: agents could be punished here for all abstaining.
            self.num_agents_participated_last = 0
            self._voter_turnout = 0
            self._update_diag_history()
            self._capture_debug_snapshot(preference_profile, aggregated=None)
            self._capture_area_snapshot_for_logger()
            return 0
        # Aggregate the preferences ⇒ returns an option ordering (indices into options)
        rule = self.model.voting_rule
        aggregated = rule(preference_profile, rng=self.model.voting_rng)
        # Save the "elected" ordering in self._voted_ordering
        winning_option = aggregated[0]
        self._voted_ordering = self.model.options[winning_option]
        # Calculate and distribute rewards
        self._distribute_rewards()

        # Adaptive participation learning update (eligible agents only)
        signals, q_pushes = self._compute_participation_learning_signals_and_q_pushes(
            self.agents, self.model
        )
        for a, signal, q_push in zip(self.agents, signals, q_pushes):
            if not a.eligible_for_election:
                continue
            if not np.isfinite(a.participation_baseline):
                a.participation_baseline = signal
            else:
                baseline = a.participation_baseline
                alpha = self.model.participation_baseline_alpha
                a.participation_baseline = (1.0 - alpha) * baseline + alpha * signal
            a.participation_signal = signal
            a.apply_participation_q_push(q_push)
        # Surprise-learning altruism update (participant-only, optional).
        if self.model.altruism_mode == "surprise_learning":
            for a in self.agents:
                if a.participating:
                    a.apply_altruism_update(a.dissatisfaction_signal)
        # Statistics
        n = preference_profile.shape[0]  # Number agents participated
        self.num_agents_participated_last = n
        area_voter_turnout = int((n / n_resident) * 100) if n_resident > 0 else 0
        self._voter_turnout = area_voter_turnout  # Update in area state
        # Logging and diagnostics
        self._update_diag_history()
        self._capture_debug_snapshot(preference_profile, aggregated)
        self._capture_area_snapshot_for_logger()
        return area_voter_turnout # Voter turnout in percent

    @staticmethod
    def _compute_participation_learning_signals(agents, model) -> list[float]:
        """Build participation-learning signals for all agents in list order.

        This function returns the diagnostic signal only.
        For the actual update payload (q-space push), see
        `_compute_participation_learning_signals_and_q_pushes`.
        """
        signals, _q_pushes = Area._compute_participation_learning_signals_and_q_pushes(
            agents, model
        )
        return signals

    @staticmethod
    def _compute_participation_learning_signals_and_q_pushes(agents, model) -> tuple[list[float], list[float]]:
        """Build participation signals and direct q-update pushes in list order.

        Modes:
        - raw_delta_rel: legacy behavior; exact pass-through of election_delta_rel.
        - group_centered_delta_rel_plus_fee: centered group mean delta_rel signal
          (eligible agents only), shrunk by group size, plus an explicit fee penalty
          for participating agents.
        - group_relative_delta_rel_party: group-level relative-performance signal
          based on mean election_delta_rel (fees already included), shrunk by group size,
          plus explicit participant fee penalty. This mode updates q directly with the
          same group-relative direction for participants and abstainers.
        """
        mode = model.participation_signal_mode
        signals = [0.0] * len(agents)
        q_pushes = [0.0] * len(agents)

        def _set_components(agent, *, group_component: float, fee_component: float) -> None:
            agent.participation_signal_group_component = group_component
            agent.participation_signal_fee_component = fee_component

        for a in agents:
            _set_components(a, group_component=0.0, fee_component=0.0)

        if mode == "raw_delta_rel":
            for idx, a in enumerate(agents):
                if not a.eligible_for_election:
                    continue
                delta_rel = a.election_delta_rel
                signals[idx] = delta_rel
                action_sign = 1.0 if bool(a.participating) else -1.0
                q_pushes[idx] = action_sign * delta_rel
                _set_components(a, group_component=delta_rel, fee_component=0.0)
            return signals, q_pushes

        if mode not in {"group_centered_delta_rel_plus_fee", "group_relative_delta_rel_party"}:
            raise ValueError(f"Unknown participation_signal_mode: {mode}")

        fee_weight = model.participation_signal_fee_weight
        shrink_k = model.participation_signal_group_shrink_k
        signal_clip = model.participation_signal_clip

        eligible_indices: list[int] = []
        group_values: dict[int, list[float]] = {}
        group_counts: dict[int, int] = {}
        for idx, a in enumerate(agents):
            if not a.eligible_for_election:
                continue
            eligible_indices.append(idx)
            g = int(a.personality_group_idx)
            delta = a.election_delta_rel
            group_values.setdefault(g, []).append(delta)
            group_counts[g] = group_counts.get(g, 0) + 1

        if not eligible_indices:
            return signals, q_pushes

        group_means = {g: np.mean(vals) for g, vals in group_values.items() if vals}
        if not group_means:
            return signals, q_pushes
        mu_groups = np.mean(list(group_means.values()))

        group_centered: dict[int, float] = {}
        for g, mu_g in group_means.items():
            n_g = group_counts.get(g, 0)
            w_g = n_g / (n_g + shrink_k) if shrink_k > 0.0 else 1.0
            group_centered[g] = w_g * (mu_g - mu_groups)

        for idx in eligible_indices:
            a = agents[idx]
            g = int(a.personality_group_idx)
            centered = group_centered.get(g, 0.0)
            fee_component = 0.0
            if a.participating:
                fee_abs = a.election_fee
                assets_post = a.assets
                delta_abs = a.election_delta_abs
                assets_pre = assets_post - delta_abs
                if np.isfinite(assets_pre) and assets_pre > 0.0 and np.isfinite(fee_abs):
                    fee_rel = max(0.0, fee_abs / assets_pre)
                    fee_component = -fee_weight * fee_rel

            signal = centered + fee_component
            signal = np.clip(signal, -signal_clip, signal_clip)
            _set_components(a, group_component=centered, fee_component=fee_component)
            signals[idx] = signal
            if mode == "group_relative_delta_rel_party":
                q_pushes[idx] = signal
            else:
                action_sign = 1.0 if bool(a.participating) else -1.0
                q_pushes[idx] = action_sign * signal
        return signals, q_pushes

    @staticmethod
    def _snapshot_agent(agent) -> dict:
        try:
            p_participation = agent.participation_probability()
        except ValueError:
            p_participation = float("nan")

        known_colors = []
        for cell in agent.known_cells:
            if cell is None:
                continue
            if isinstance(cell, (int, np.integer)):
                known_colors.append({"color": int(cell)})
            else:
                known_colors.append({"color": int(cell.color)})

        personality_group = agent.personality_group
        if isinstance(personality_group, np.ndarray):
            personality_group = personality_group.tolist()

        personality = agent.personality
        if isinstance(personality, np.ndarray):
            personality = personality.tolist()

        est_real_dist = agent.est_real_dist
        if isinstance(est_real_dist, np.ndarray):
            est_real_dist = est_real_dist.tolist()

        delta_abs = agent.election_delta_abs
        assets_now = agent.assets

        return {
            "id": agent.unique_id,
            "pos": agent.position,
            "personality_group_idx": agent.personality_group_idx,
            "personality_group": personality_group,
            "personality": personality,
            "assets": assets_now,
            "assets_pre_est": assets_now - delta_abs,
            "eligible": bool(agent.eligible_for_election),
            "participating": bool(agent.participating),
            "num_elections_participated": agent.num_elections_participated,
            "fee": agent.election_fee,
            "reward_personal": agent.reward_personal,
            "delta_abs": delta_abs,
            "delta_rel": agent.election_delta_rel,
            "q_participation": agent.q_participation,
            "p_participation": p_participation,
            "altruism_factor": agent.altruism_factor,
            "dissatisfaction_value": agent.dissatisfaction_value,
            "dissatisfaction_baseline": agent.dissatisfaction_baseline,
            "dissatisfaction_signal": agent.dissatisfaction_signal,
            "est_real_dist": est_real_dist,
            "confidence": agent.confidence,
            "known_cells_count": len(known_colors),
            "known_cells": known_colors,
            "award_history_tail": list(agent.award_history[-5:]),
            "participation_strategy": agent.participation_strategy.__class__.__name__,
            "voting_strategy": agent.voting_strategy.__class__.__name__,
        }

    def _tally_votes(self) -> np.ndarray:
        """
        Gathers votes from agents who choose to participate.

        Each participating agent contributes a ScoreVector of oppose-scores over
        the available options (lower = better). These are stacked into a matrix.

        Returns:
            np.ndarray: 2D array where each row is an agent's ScoreVector
            and each column corresponds to an option.
        """
        preference_profile = []
        debug_enabled = self._debug_enabled()
        debug_votes = [] if debug_enabled else None
        # Reset pool for this election step.
        self._election_fee_pool = 0
        el_cost_rate = self.model.election_cost_rate

        # Optional output vote sink: logger attaches a callable here.
        vote_sink = self.model._output_vote_sink

        for agent in self.agents:
            # Reset per-election asset delta signal for learning.
            agent.reset_reward_variables()
            # Eligibility: agents with assets <= 0 are skipped (no learning update).
            if agent.assets <= 0:
                agent.mark_ineligible_for_election()
                continue

            # election_cost_rate is a fraction (0..1) of current assets.
            cost = agent.assets * el_cost_rate
            if cost < 0:
                raise ValueError("Election cost rate must be non-negative.")

            if agent.ask_for_participation(area=self):
                agent.mark_participating()
                agent.num_elections_participated += 1
                # Collect the participation _fee into the area pool
                agent.set_election_fee(cost)  # Fee will be applied when rewards are distributed
                self._election_fee_pool += cost
                # Ask the agent for her preference
                scores = np.asarray(agent.vote(area=self), dtype=np.float32)
                # Representation contract: VoteAgent.vote returns a 1D ScoreVector
                # over *options* (not colors). Fail fast instead of silently
                # treating malformed votes as "no participants".
                validate_score_vector_unit_interval(scores, int(self.model.options.shape[0]))
                preference_profile.append(scores)

                # Emit participant vote context if a sink is configured.
                if vote_sink is not None:
                    vote_sink(
                        area=self,
                        agent=agent,
                        oppose_scores=scores,
                        est_dist=agent.est_real_dist,
                        confidence=agent.confidence,
                        voted_altruistically=agent.voted_altruistically,
                    )
                if debug_enabled:
                    scores = np.asarray(scores, dtype=np.float64)
                    debug_rng = self.model.rng_debug
                    ordering = scores_to_ordering(scores, rng=debug_rng).tolist()
                    debug_votes.append(
                        {
                            "agent_id": agent.unique_id,
                            "scores": scores.tolist(),
                            "ordering": ordering,
                        }
                    )
                # agent.vote returns a ScoreVector (oppose scores) for each option
        if debug_enabled:
            self._debug_last_votes = debug_votes or []
        else:
            self._debug_last_votes = []
        return np.array(preference_profile)

    def _distribute_rewards(self) -> None:
        """
        Calculates and distributes per-election rewards/punishments.

        Binary quality-sign contract:
        - Decision quality gate:
            good if quality_distance <= break_even_distance_common, else bad
            quality_distance is selected by model.quality_target_mode:
                - "reality": dist_to_reality
                - "puzzle":  puzzle_distance
        - Unified reward rate:
            reward_rate_personal (fraction of current assets)
        - Group distance factor:
            group_dst_to_outcome = dist(personality_group, voted_ordering) in [0,1]
        - Reward rule:
            if good: reward = +(1 - group_dst_to_outcome) * reward_rate * assets_pre
            if bad:  reward = -(group_dst_to_outcome)     * reward_rate * assets_pre
        - Election fee remains separate and participant-only in _tally_votes()/reward_agent().
        """
        self._update_quality_distances()
        self._update_group_outcome_distance()
        quality_distance = self._quality_distance()

        quality_threshold = self.model.break_even_distance_common
        decision_good = quality_distance <= quality_threshold + 1e-12
        sign = 1.0 if decision_good else -1.0
        reward_rate = self.model.reward_rate_personal

        for a in self.agents:
            # group_dst_to_outcome: canonical personality-group distance to elected ordering
            g = int(a.personality_group_idx)
            if 0 <= g < len(self._group_outcome_distance):
                group_dst_to_outcome = self._group_outcome_distance[g]
            else:
                group_dst_to_outcome = float("nan")
            if not np.isfinite(group_dst_to_outcome):
                group_dst_to_outcome = 1.0
            if decision_good:
                reward_factor = 1.0 - group_dst_to_outcome
            else:
                reward_factor = group_dst_to_outcome
            reward_amount = sign * reward_rate * reward_factor * a.assets

            a.add_personal_reward(reward_amount)
            a.reward_agent()  # Apply accumulated rewards/penalties to assets (and store delta signals)

    def _compute_group_outcome_distance(self) -> list[float]:
        dist_func = self.model.distance_func
        search_pairs = self.model.color_search_pairs
        voted_ordering = self.voted_ordering
        num_groups = int(len(self.model.personality_groups))
        if voted_ordering is None or num_groups <= 0:
            return [float("nan")] * max(0, num_groups)
        out = [float("nan")] * num_groups
        for g in range(num_groups):
            group_ordering = np.asarray(self.model.personality_groups[g], dtype=np.int64)
            d = dist_func(group_ordering, voted_ordering, search_pairs)
            out[g] = float(np.clip(d, 0.0, 1.0))
        return out

    def _update_group_outcome_distance(self) -> None:
        self._group_outcome_distance = self._compute_group_outcome_distance()

    @staticmethod
    def _ordering_from_distribution_tie_aware(
            dist: np.ndarray,
            *,
            reference_ordering: np.ndarray | None,
            rng: np.random.Generator | None,
    ) -> np.ndarray:
        """Compatibility wrapper; canonical implementation lives in utils.representations."""
        return distribution_to_ordering_tie_aware(
            dist,
            reference_ordering=reference_ordering,
            rng=rng,
            atol=1e-12,
            rtol=0.0,
        )

    def _ordering_to_option_id(self, ordering: np.ndarray | None) -> int:
        if ordering is None:
            return -1
        return int(self.model.option_id_for_ordering(ordering))

    def update_color_distribution(self) -> None:
        """
        Recalculates the area's color distribution and updates the _color_distribution attribute.

        This method counts how many cells of each color belong to the area, normalizes
        the counts by the total number of cells, and stores the result internally.
        """
        counts = np.zeros(self.model.num_colors, dtype=np.int64)
        for cell in self.cells:
            counts[int(cell.color)] += 1
        self._color_counts = counts
        if self.num_cells > 0:
            self._color_distribution = counts.astype(np.float64) / self.num_cells

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

    def _normalize_puzzle_distribution(self, sample: np.ndarray) -> np.ndarray:
        """Normalize puzzle samples without edge hacks; fallback to uniform if invalid."""
        num_colors = self.model.num_colors
        if num_colors <= 0:
            return np.asarray([], dtype=np.float64)
        arr = np.asarray(sample, dtype=np.float64)
        if arr.shape != (num_colors,) or not np.all(np.isfinite(arr)):
            return np.full(num_colors, 1.0 / num_colors, dtype=np.float64)
        arr = np.clip(arr, 0.0, None)
        total = arr.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.full(num_colors, 1.0 / num_colors, dtype=np.float64)
        return arr / total

    def _sample_fresh_puzzle_distribution(self) -> np.ndarray:
        """Sample a color-symmetric, center-biased puzzle distribution on the simplex."""
        num_colors = self.model.num_colors
        if num_colors <= 0:
            return np.asarray([], dtype=np.float64)
        alpha = np.full(num_colors, _PUZZLE_ALPHA_SHOCK_PER_COMPONENT, dtype=np.float64)
        sample = np.asarray(self.model.rng_puzzle.dirichlet(alpha), dtype=np.float64)
        if sample.shape != (num_colors,) or not np.all(np.isfinite(sample)):
            return np.full(num_colors, 1.0 / num_colors, dtype=np.float64)
        return self._normalize_puzzle_distribution(sample)

    def _sample_local_puzzle_distribution(self, prev: np.ndarray) -> np.ndarray:
        """Sample a local random walk step around the previous puzzle distribution."""
        prev_arr = np.asarray(prev, dtype=np.float64)
        num_colors = self.model.num_colors
        if prev_arr.shape != (num_colors,) or not np.all(np.isfinite(prev_arr)):
            return self._sample_fresh_puzzle_distribution()
        prev_arr = np.clip(prev_arr, 0.0, None)
        total = prev_arr.sum()
        if not np.isfinite(total) or total <= 0.0:
            return self._sample_fresh_puzzle_distribution()
        prev_arr = prev_arr / total
        alpha = self.model.puzzle_local_kappa * prev_arr + _PUZZLE_ALPHA_CENTER_PER_COMPONENT
        sample = np.asarray(self.model.rng_puzzle.dirichlet(alpha), dtype=np.float64)
        if sample.shape != (num_colors,) or not np.all(np.isfinite(sample)):
            return self._sample_fresh_puzzle_distribution()
        return self._normalize_puzzle_distribution(sample)

    def _update_puzzle_distribution(self) -> None:
        """Update area-level puzzle distribution via Dirichlet RW + rare shocks."""
        prev = self._puzzle_distribution
        if prev is None:
            self._puzzle_distribution = self._sample_fresh_puzzle_distribution()
            return

        shock_prob = self.model.puzzle_shock_prob
        if shock_prob >= 1.0 or (shock_prob > 0.0 and self.model.rng_puzzle.random() < shock_prob):
            self._puzzle_distribution = self._sample_fresh_puzzle_distribution()
            return
        self._puzzle_distribution = self._sample_local_puzzle_distribution(prev)

    def _clear_puzzle_state(self) -> None:
        """Drop per-step puzzle state when running in reality mode."""
        self._puzzle_distribution = None
        self._puzzle_distance = float("nan")
        self._puzzle_ordering_for_quality = None

    def _compute_puzzle_distance(self) -> float:
        puzzle_distribution = self._puzzle_distribution
        if puzzle_distribution is None or puzzle_distribution.size == 0:
            self._puzzle_ordering_for_quality = None
            return float("nan")
        puzzle_ord = self._ordering_from_distribution_tie_aware(
            np.asarray(puzzle_distribution, dtype=np.float64),
            reference_ordering=self.voted_ordering,
            rng=self.model.rng_puzzle,
        )
        self._puzzle_ordering_for_quality = np.asarray(puzzle_ord, dtype=np.int64)
        return self.model.distance_func(puzzle_ord, self.voted_ordering, self.model.color_search_pairs)

    def _update_quality_distances(self) -> None:
        voted_ordering = self.voted_ordering
        real_color_ord = self._ordering_from_distribution_tie_aware(
            self.color_distribution,
            reference_ordering=voted_ordering,
            rng=self.model.voting_rng,
        )
        self._grid_ordering_for_quality = np.asarray(real_color_ord, dtype=np.int64)
        self._dist_to_reality = self.model.distance_func(real_color_ord, voted_ordering, self.model.color_search_pairs)
        if self.puzzle_mode and self._puzzle_distribution is None:
            self._update_puzzle_distribution()
        self._puzzle_distance = self._compute_puzzle_distance()

    def _quality_distance(self) -> float:
        if self.puzzle_mode:
            quality_distance = self.puzzle_distance
            if not np.isfinite(quality_distance):
                raise RuntimeError("quality_target_mode='puzzle' requires finite puzzle_distance.")
            return quality_distance
        return self.dist_to_reality

    def _update_diag_history(self) -> None:
        """Append per-area diagnostics for the current step."""
        agents = [a for a in self.agents if a is not None]
        eligible = [a for a in agents if a.eligible_for_election]
        participants = [a for a in eligible if a.participating]
        abstainers = [a for a in eligible if not a.participating]

        def _mean(vals):
            return np.mean(vals) if len(vals) > 0 else float("nan")

        def _mean_attr(pool, attr):
            return _mean([getattr(a, attr) for a in pool])

        def _mean_participation_prob(pool):
            vals = []
            for a in pool:
                try:
                    vals.append(a.participation_probability())
                except ValueError:
                    vals.append(float("nan"))
            return _mean([v for v in vals if np.isfinite(v)])

        # Per-election delta (relative) means
        mean_delta_rel_participants = _mean_attr(participants, "election_delta_rel")
        mean_delta_rel_abstainers = _mean_attr(abstainers, "election_delta_rel")

        # Reward component (absolute)
        mean_personal_reward = _mean_attr(eligible, "reward_personal")

        # Learning signals
        mean_q_participation = _mean_attr(eligible, "q_participation")
        mean_p_participation = _mean_participation_prob(eligible)
        mean_altruism = _mean_attr(eligible, "altruism_factor")

        # Area gini (0-100)
        from src.utils.metrics import gini_index_0_100
        assets = [a.assets for a in agents]
        gini = int(gini_index_0_100(assets)) if assets else 0

        # Per-personality_group metrics (area-level)
        pg = self.model.personality_groups
        num_groups = len(pg)
        group_turnout = [float("nan")] * num_groups
        group_mean_assets = [float("nan")] * num_groups
        group_mean_delta_rel = [float("nan")] * num_groups
        group_mean_delta_rel_participants = [float("nan")] * num_groups
        group_mean_delta_rel_abstainers = [float("nan")] * num_groups
        group_mean_personal_reward = [float("nan")] * num_groups
        group_mean_fee = [float("nan")] * num_groups
        group_mean_altruism = [float("nan")] * num_groups
        group_mean_q_participation_participants = [float("nan")] * num_groups
        group_mean_q_participation_abstainers = [float("nan")] * num_groups
        group_mean_dissatisfaction = [float("nan")] * num_groups

        if num_groups > 0:
            for g in range(num_groups):
                g_agents = [a for a in agents if int(a.personality_group_idx) == g]
                g_eligible = [a for a in eligible if int(a.personality_group_idx) == g]
                g_participants = [a for a in g_eligible if a.participating]
                g_abstainers = [a for a in g_eligible if not a.participating]

                if g_eligible:
                    group_turnout[g] = len(g_participants) / len(g_eligible) * 100.0
                    group_mean_delta_rel[g] = _mean_attr(g_eligible, "election_delta_rel")
                    group_mean_delta_rel_participants[g] = _mean_attr(g_participants, "election_delta_rel")
                    group_mean_delta_rel_abstainers[g] = _mean_attr(g_abstainers, "election_delta_rel")
                    group_mean_q_participation_participants[g] = _mean_attr(g_participants, "q_participation")
                    group_mean_q_participation_abstainers[g] = _mean_attr(g_abstainers, "q_participation")
                if g_agents:
                    group_mean_assets[g] = _mean_attr(g_agents, "assets")
                    group_mean_personal_reward[g] = _mean_attr(g_agents, "reward_personal")
                    group_mean_fee[g] = _mean_attr(g_agents, "election_fee")
                    group_mean_altruism[g] = _mean_attr(g_agents, "altruism_factor")
                    group_mean_dissatisfaction[g] = _mean_attr(g_agents, "dissatisfaction_value")

        self._diag_history.append(
            {
                "turnout": self.voter_turnout,
                "dist_to_reality": self.dist_to_reality,
                "puzzle_distance": self.puzzle_distance,
                "group_outcome_distance": list(self.group_outcome_distance),
                "mean_delta_rel_participants": mean_delta_rel_participants,
                "mean_delta_rel_abstainers": mean_delta_rel_abstainers,
                "mean_personal_reward": mean_personal_reward,
                "mean_q_participation": mean_q_participation,
                "mean_p_participation": mean_p_participation,
                "mean_altruism": mean_altruism,
                "gini": gini,
                "group_turnout": group_turnout,
                "group_mean_assets": group_mean_assets,
                "group_mean_delta_rel": group_mean_delta_rel,
                "group_mean_delta_rel_participants": group_mean_delta_rel_participants,
                "group_mean_delta_rel_abstainers": group_mean_delta_rel_abstainers,
                "group_mean_personal_reward": group_mean_personal_reward,
                "group_mean_fee": group_mean_fee,
                "group_mean_altruism": group_mean_altruism,
                "group_mean_q_participation_participants": group_mean_q_participation_participants,
                "group_mean_q_participation_abstainers": group_mean_q_participation_abstainers,
                "group_mean_dissatisfaction": group_mean_dissatisfaction,
            }
        )

    def _debug_enabled(self) -> bool:
        return bool(getattr(self.model, "_debug_agent_panel_enabled", False))

    def _debug_max_steps(self) -> int:
        max_steps = int(getattr(self.model, "_debug_agent_panel_max_steps", 1))
        return max(1, max_steps)

    def _capture_debug_snapshot(self, pref_profile: np.ndarray, aggregated) -> None:
        """Capture a detailed snapshot of the area's state for debugging purposes."""
        if not self._debug_enabled():
            return

        step = int(self.model.scheduler.steps)
        agents = sorted(self.agents, key=lambda a: int(a.unique_id))
        eligible = [a for a in agents if a.eligible_for_election]
        participants = [a for a in eligible if a.participating]
        abstainers = [a for a in eligible if not a.participating]

        # If no one participates, `aggregated` is None and the UI may show
        # winning_option=None. However, the simulation can still have a carried-over
        # `voted_ordering`. Compute the corresponding option id for clarity.
        winning_option_id = None
        try:
            vo = self._voted_ordering
            if vo is not None:
                oid = int(self.model.option_id_for_ordering(vo))
                if oid >= 0:
                    winning_option_id = oid
        except ValueError:
            winning_option_id = None

        record = {
            "step": step,
            "area_id": self.unique_id,
            "num_agents": len(agents),
            "num_eligible": len(eligible),
            "num_participants": len(participants),
            "num_abstainers": len(abstainers),
            "election_held": bool(aggregated is not None),
            "winning_option_id": winning_option_id,
            "dist_to_reality": self._dist_to_reality,
            "puzzle_distance": self._puzzle_distance,
            "quality_target_mode": self.model.quality_target_mode,
            "quality_distance_source": "puzzle_distance" if self.puzzle_mode else "dist_to_reality",
            "quality_distance": self._quality_distance(),
            "voted_ordering": (
                self._voted_ordering.tolist()
                if isinstance(self._voted_ordering, np.ndarray)
                else list(self._voted_ordering)
                if self._voted_ordering is not None
                else None
            ),
            "real_color_distribution": (
                self.color_distribution.tolist()
                if isinstance(self.color_distribution, np.ndarray)
                else list(self.color_distribution)),
            "preference_profile": (
                pref_profile.tolist() if isinstance(pref_profile, np.ndarray)
                else list(pref_profile)
            ),
            "votes": list(self._debug_last_votes or []),
            "aggregated_ordering": (
                aggregated.tolist()
                if isinstance(aggregated, np.ndarray)
                else list(aggregated)
                if aggregated is not None
                else None
            ),
            "winning_option": int(
                aggregated[0]) if aggregated is not None and len(
                aggregated) > 0 else None,
            "quality_threshold_common": self.model.break_even_distance_common,
            "reward_rate": self.model.reward_rate_personal,
            "agents": [self._snapshot_agent(a) for a in agents],
        }

        self._debug_history.append(record)
        max_steps = self._debug_max_steps()
        if len(self._debug_history) > max_steps:
            self._debug_history = self._debug_history[-max_steps:]

    def _capture_area_snapshot_for_logger(self) -> None:
        # --- optional output hook: snapshot right before mutation (pre-mutation)
        snapshot_sink = self.model._output_area_snapshot_sink
        if snapshot_sink is not None:
            from src.utils.metrics import gini_index_0_100

            fee_pool = self.election_fee_pool
            eligible_voters = self.num_eligible_voters_last
            participants = self.num_agents_participated_last
            turnout = self.voter_turnout
            dist_to_reality = self.dist_to_reality
            puzzle_distance = self.puzzle_distance
            assets = [a.assets for a in self.agents]
            gini_index = int(gini_index_0_100(assets)) if assets else 0
            area_color = None
            if self.color_distribution is not None:
                area_color = self.color_distribution.copy()
            elected_color = None
            if self.voted_ordering is not None:
                elected_color = self.voted_ordering.copy()
            grid_ordering_id = self._ordering_to_option_id(self._grid_ordering_for_quality)
            puzzle_ordering_id = self._ordering_to_option_id(self._puzzle_ordering_for_quality)
            snapshot_sink(
                area=self,
                snapshot={
                    "fee_pool": fee_pool,
                    "eligible_voters": eligible_voters,
                    "participants": participants,
                    "turnout": turnout,
                    "dist_to_reality": dist_to_reality,
                    "puzzle_distance": puzzle_distance if puzzle_distance is not None else float("nan"),
                    "gini_index": gini_index,
                    "area_color": area_color,
                    "elected_color": elected_color,
                    "group_outcome_distance": list(self.group_outcome_distance),
                    "grid_ordering_id": int(grid_ordering_id),
                    "puzzle_ordering_id": int(puzzle_ordering_id),
                    "puzzle_color": (
                        self.puzzle_distribution.copy()
                        if isinstance(self.puzzle_distribution, np.ndarray)
                        else None
                    ),
                },
            )

    def mutate_cells(self) -> None:
        """Mutate cell colors based on the last election outcome."""
        if self.voter_turnout == 0:
            return
        # Take some number of cells to mutate (i.e., 5 %)
        n_to_mutate = int(self.model.mu * self.num_cells)
        # Idea for future: What if the voter_turnout determines the mutation rate?
        cells_to_mutate = self.model.random.sample(self.cells, n_to_mutate)
        # Use voted ordering to pick colors in descending order
        # To pre-select colors for all cells to mutate
        colors = self.model.np_random.choice(self.voted_ordering,
                                             size=n_to_mutate,
                                             p=self.model.color_probs)
        # Assign the newly selected colors to the cells
        for cell, color in zip(cells_to_mutate, colors):
            cell.color = color
        # Important: Update the color distribution (because colors changed)
        if not self.model.no_overlap:  # There may be overlap
            print("Warning: there may be overlapping areas; color distribution has to be updated in accordance.")
            return
        self.update_color_distribution()

    def step(self) -> None:
        """
        Run one step of the simulation.

        Conduct an election in the area,
        mutate the cells' colors according to the election outcome
        and update the color distribution of the area.
        """
        if self.puzzle_mode:
            self._update_puzzle_distribution()
        else:
            self._clear_puzzle_state()

        # Update knowledge for all agents before any learning/election logic.
        for agent in self.agents:
            agent.update_known_cells(area=self)
            # Dissatisfaction is computed from current (pre-election) distributions.
            # Baseline is EMA; if alpha=1.0, signal equals last-step delta.
            sv = agent.compute_dissatisfaction_value(area=self, model=self.model)
            agent.dissatisfaction_value = sv
            if not np.isfinite(agent.dissatisfaction_baseline):
                # Initialize baseline on first observation to avoid a large spike.
                agent.dissatisfaction_baseline = sv
                agent.dissatisfaction_signal = 0.0
            else:
                baseline = agent.dissatisfaction_baseline
                agent.dissatisfaction_signal = sv - baseline
                alpha = self.model.satisfaction_baseline_alpha
                agent.dissatisfaction_baseline = (1.0 - alpha) * baseline + alpha * sv
        if self.model.altruism_mode == "satisfaction":
            for agent in self.agents:
                agent.apply_altruism_satisfaction_mode(agent.dissatisfaction_value)
        self.conduct_election()
        # self.mutate_cells()
