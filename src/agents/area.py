from __future__ import annotations
from typing import TYPE_CHECKING, cast, List
import numpy as np
from mesa import Agent
if TYPE_CHECKING:  # Type hint for IDEs
    from src.models.participation_model import ParticipationModel
    from src.agents.color_cell import ColorCell
    from src.agents.vote_agent import VoteAgent
from src.utils.representations import scores_to_ordering
from src.utils.rng import np_rng_debug


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
        self.np_random = model.random  # Use the model's random generator for reproducibility
        self._set_dimensions(width, height, size_variance)
        self.agents: List["VoteAgent"] = []
        self._personality_group_distribution = None
        self.cells: List["ColorCell"] = []
        self._idx_field = None  # An indexing position of the area in the grid
        self._color_distribution = np.zeros(model.num_colors) # Initialize to 0
        self._voted_ordering = None
        self._voter_turnout = 0  # In percent
        self._dist_to_reality = None  # Elected vs. actual color distribution
        self._election_fee_pool: float = 0
        self._num_agents_participated_last = None  # For statistics
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
        This method calculates the areas current distribution of personality groups.
        """
        personality_groups = list(self.model.personality_groups)
        p_counts = {str(i): 0 for i in personality_groups}
        # Count the occurrence of each personality_group (color ordering)
        for agent in self.agents:
            p_counts[str(agent.personality_group)] += 1
        # Normalize the counts
        if self.num_agents == 0:
            self._personality_group_distribution = [0 for _ in personality_groups]
        else:
            self._personality_group_distribution = [p_counts[str(p)] / self.num_agents
                                              for p in personality_groups]

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
            int: The voter turnout in percent. Returns 0 if no agent participates.
        """
        # Ask agents for participation and their votes
        preference_profile = self._tally_votes()
        # Check for the case that no agent participated
        if preference_profile.ndim != 2 or preference_profile.shape[0] == 0:
            # Set to previous outcome but don't distribute rewards as usual
            print("Area", self.unique_id, "no one participated in the election")
            # If no previous outcome, use the real distribution ordering
            real_color_ord = np.argsort(self.color_distribution, kind="stable")[::-1]
            if self._voted_ordering is None:
                self._voted_ordering = real_color_ord
            # Update dist_to_reality for monitoring but no rewards
            self._dist_to_reality = self.model.distance_func(
                real_color_ord, self._voted_ordering,
                self.model.color_search_pairs
            )
            # Slightly punish for non-participation
            for a in self.agents:
                a.add_common_reward(-1)
                a.reward_agent()  # Apply the penalty
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
        for a in self.agents:
            # Eligible agents are exactly those evaluated in _tally_votes()
            if a.eligible_for_election:
                a.apply_participation_update(a.election_delta_rel)
        # TODO put those two loops together
        # Adaptive altruism learning update (participant-only)
        for a in self.agents:
            if a.participating:
                a.apply_altruism_update(a.election_delta_rel)
        # Statistics
        n = preference_profile.shape[0]  # Number agents participated
        self.num_agents_participated_last = n
        area_voter_turnout = int((n / self.num_agents) * 100)
        self._voter_turnout = area_voter_turnout  # Update in area state
        # Logging and diagnostics
        self._update_diag_history()
        self._capture_debug_snapshot(preference_profile, aggregated)
        self._capture_area_snapshot_for_logger()
        return area_voter_turnout # Voter turnout in percent

    @staticmethod
    def _snapshot_agent(agent) -> dict:
        try:
            p_participation = float(agent.participation_probability())
        except ValueError:
            p_participation = float("nan")

        known_colors = []
        for cell in getattr(agent, "known_cells", []) or []:
            if cell is None:
                continue
            known_colors.append({"color": getattr(cell, "color", None),})

        personality_group = getattr(agent, "personality_group", None)
        if isinstance(personality_group, np.ndarray):
            personality_group = personality_group.tolist()

        personality = getattr(agent, "personality", None)
        if isinstance(personality, np.ndarray):
            personality = personality.tolist()

        est_real_dist = getattr(agent, "est_real_dist", None)
        if isinstance(est_real_dist, np.ndarray):
            est_real_dist = est_real_dist.tolist()

        delta_abs = float(getattr(agent, "election_delta_abs", float("nan")))
        assets_now = float(getattr(agent, "assets", float("nan")))

        return {
            "id": getattr(agent, "unique_id", None),
            "pos": getattr(agent, "position", None),
            "personality_group_idx": getattr(agent, "personality_group_idx", None),
            "personality_group": personality_group,
            "personality": personality,
            "assets": assets_now,
            "assets_pre_est": assets_now - delta_abs,
            "eligible": bool(getattr(agent, "eligible_for_election", False)),
            "participating": bool(getattr(agent, "participating", False)),
            "num_elections_participated": getattr(agent, "num_elections_participated", None),
            "fee": float(getattr(agent, "_fee", 0.0)),
            "reward_common": float(getattr(agent, "_reward_common_comp", 0.0)),
            "reward_personal": float(getattr(agent, "_reward_pers_comp", 0.0)),
            "delta_abs": delta_abs,
            "delta_rel": float(getattr(agent, "election_delta_rel", float("nan"))),
            "q_participation": float(getattr(agent, "q_participation", float("nan"))),
            "p_participation": p_participation,
            "altruism_factor": float(getattr(agent, "altruism_factor", float("nan"))),
            "est_real_dist": est_real_dist,
            "confidence": float(getattr(agent, "confidence", float("nan"))),
            "known_cells_count": len(known_colors),
            "known_cells": known_colors,
            "award_history_tail": list(getattr(agent, "award_history", [])[-5:]),
            "participation_strategy": getattr(
                getattr(agent, "participation_strategy", None), "__class__", type("X", (), {})
            ).__name__,
            "voting_strategy": getattr(
                getattr(agent, "voting_strategy", None), "__class__", type("X", (), {})
            ).__name__,
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

        # Optional schema-v2 vote sink (Batch 2): logger attaches a callable here.
        vote_sink = getattr(self.model, "_schema_v2_vote_sink", None)

        for agent in self.agents:
            # Reset per-election asset delta signal for learning.
            agent.reset_reward_variables()
            # Eligibility: agents with assets <= 0 are skipped (no learning update).
            if agent.assets <= 0:
                agent.mark_ineligible_for_election()
                continue

            # election_cost_rate is treated as a percent (0..100) of current assets.
            cost = int(agent.assets * el_cost_rate)
            # Ensure participating agents pay at least 1 if they have assets.
            if cost == 0 and agent.assets >= 1 and not el_cost_rate == 0:
                cost = 1

            if agent.ask_for_participation(area=self):
                agent.mark_participating()
                agent.num_elections_participated += 1
                # Give agents their (new) known fields
                agent.update_known_cells(area=self)
                # Collect the participation _fee into the area pool
                agent.set_election_fee(cost)  # Fee will be applied when rewards are distributed
                self._election_fee_pool += cost
                # Ask the agent for her preference
                scores = agent.vote(area=self)
                preference_profile.append(scores)

                # Emit participant vote context if a sink is configured.
                if vote_sink is not None:
                    vote_sink(
                        area=self,
                        agent=agent,
                        oppose_scores=scores,
                        est_dist=getattr(agent, "est_real_dist", None),
                        confidence=getattr(agent, "confidence", None),
                    )
                if debug_enabled:
                    scores = np.asarray(scores, dtype=np.float64)
                    debug_rng = getattr(self.model, "rng_debug", np_rng_debug())
                    ordering = scores_to_ordering(scores, rng=debug_rng).astype(int).tolist()
                    debug_votes.append(
                        {
                            "agent_id": getattr(agent, "unique_id", None),
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
        Calculates and distributes rewards (or penalties) to agents based on outcomes.

        Contract (economics v2):
        - Signs are determined by distances in [0,1] mapped via (threshold - d)
        - Magnitudes are scaled by agent wealth via model.reward_rate_* (0..1)
        - Fee pool is tracked as a statistic but no longer sets reward magnitude
        """
        dist_func = self.model.distance_func
        # Calculate the distance to the real distribution using distance_func in [0,1]
        real_color_ord = np.argsort(self.color_distribution, kind="stable")[::-1]  # Descending
        search_pairs = self.model.color_search_pairs
        self._dist_to_reality = dist_func(
            real_color_ord, self.voted_ordering, search_pairs
        )
        # Common component coefficient shared across agents
        common_coeff = (self.model.reward_threshold_common - float(self.dist_to_reality))
        # Model-wide reward rates for scaling rewards/penalties by agent wealth
        reward_rate_common = self.model.reward_rate_common
        reward_rate_personal = self.model.reward_rate_personal
        abstention_share = self.model.abstention_share
        for a in self.agents:
            # Personality-based reward factor
            #   the closer the elected outcome to the agent's personality_group.
            #   the higher the reward for the agent.
            # TODO(thesis): later switch this to a centralized distribution-distance
            #   between a.personal_opt_dist (agent personality dist) and the elected outcome
            #   expressed as a distribution (not ordering).
            p = dist_func(a.personality_group, self.voted_ordering, search_pairs)
            pers_coeff = (self.model.reward_threshold_personal - p)

            # Absolute rewards/penalties in asset units
            scale_common = reward_rate_common * a.assets  # Scale by current wealth
            scale_personal = reward_rate_personal * a.assets
            pers_component = pers_coeff * scale_personal
            common_component = common_coeff * scale_common
            if not a.participating:
                common_component *= abstention_share
            # Save and apply rewards/penalties to the agent.
            a.add_personal_reward(pers_component)
            a.add_common_reward(common_component)
            a.reward_agent()  # Apply accumulated rewards/penalties to assets (and store delta signals)

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

    def _update_diag_history(self) -> None:
        """Append per-area diagnostics for the current step."""
        agents = [a for a in self.agents if a is not None]
        eligible = [a for a in agents if bool(getattr(a, "eligible_for_election", False))]
        participants = [a for a in eligible if bool(getattr(a, "participating", False))]
        abstainers = [a for a in eligible if not bool(getattr(a, "participating", False))]

        def _mean(vals):
            return float(np.mean(vals)) if len(vals) > 0 else float("nan")

        def _mean_attr(pool, attr):
            return _mean([float(getattr(a, attr, 0.0)) for a in pool])

        def _mean_participation_prob(pool):
            vals = []
            for a in pool:
                try:
                    vals.append(float(a.participation_probability()))
                except ValueError:
                    vals.append(float("nan"))
            return _mean([v for v in vals if np.isfinite(v)])

        # Per-election delta (relative) means
        mean_delta_rel_participants = _mean_attr(participants, "election_delta_rel")
        mean_delta_rel_abstainers = _mean_attr(abstainers, "election_delta_rel")

        # Reward components (absolute)
        mean_common_reward = _mean_attr(eligible, "_reward_common_comp")
        mean_personal_reward = _mean_attr(eligible, "_reward_pers_comp")

        # Learning signals
        mean_q_participation = _mean_attr(eligible, "q_participation")
        mean_p_participation = _mean_participation_prob(eligible)
        mean_altruism = _mean_attr(eligible, "altruism_factor")

        # Area gini (0-100)
        from src.utils.metrics import gini_index_0_100
        assets = [float(getattr(a, "assets", 0.0)) for a in eligible]
        gini = int(gini_index_0_100(assets)) if assets else 0

        # Per-personality_group metrics (area-level)
        pg = getattr(self.model, "personality_groups", None)
        num_groups = len(pg) if pg is not None else 0
        group_turnout = [float("nan")] * num_groups
        group_mean_assets = [float("nan")] * num_groups
        group_mean_delta_rel = [float("nan")] * num_groups
        group_mean_delta_rel_participants = [float("nan")] * num_groups
        group_mean_delta_rel_abstainers = [float("nan")] * num_groups
        group_mean_common_reward = [float("nan")] * num_groups
        group_mean_personal_reward = [float("nan")] * num_groups
        group_mean_fee = [float("nan")] * num_groups
        group_mean_altruism = [float("nan")] * num_groups
        group_mean_q_participation_participants = [float("nan")] * num_groups
        group_mean_q_participation_abstainers = [float("nan")] * num_groups

        if num_groups > 0:
            for g in range(num_groups):
                g_agents = [a for a in agents if int(getattr(a, "personality_group_idx", -1)) == g]
                g_eligible = [a for a in eligible if int(getattr(a, "personality_group_idx", -1)) == g]
                g_participants = [a for a in g_eligible if bool(getattr(a, "participating", False))]
                g_abstainers = [a for a in g_eligible if not bool(getattr(a, "participating", False))]

                if g_eligible:
                    group_turnout[g] = float(len(g_participants) / len(g_eligible) * 100.0)
                    group_mean_delta_rel[g] = _mean_attr(g_eligible, "election_delta_rel")
                    group_mean_delta_rel_participants[g] = _mean_attr(g_participants, "election_delta_rel")
                    group_mean_delta_rel_abstainers[g] = _mean_attr(g_abstainers, "election_delta_rel")
                    group_mean_q_participation_participants[g] = _mean_attr(g_participants, "q_participation")
                    group_mean_q_participation_abstainers[g] = _mean_attr(g_abstainers, "q_participation")
                if g_agents:
                    group_mean_assets[g] = _mean_attr(g_agents, "assets")
                    group_mean_common_reward[g] = _mean_attr(g_agents, "_reward_common_comp")
                    group_mean_personal_reward[g] = _mean_attr(g_agents, "_reward_pers_comp")
                    group_mean_fee[g] = _mean_attr(g_agents, "_fee")
                    group_mean_altruism[g] = _mean_attr(g_agents, "altruism_factor")

        self._diag_history.append(
            {
                "turnout": float(self.voter_turnout),
                "dist_to_reality": float(self.dist_to_reality) if self.dist_to_reality is not None else float("nan"),
                "mean_delta_rel_participants": mean_delta_rel_participants,
                "mean_delta_rel_abstainers": mean_delta_rel_abstainers,
                "mean_common_reward": mean_common_reward,
                "mean_personal_reward": mean_personal_reward,
                "mean_q_participation": mean_q_participation,
                "mean_p_participation": mean_p_participation,
                "mean_altruism": mean_altruism,
                "gini": float(gini),
                "group_turnout": group_turnout,
                "group_mean_assets": group_mean_assets,
                "group_mean_delta_rel": group_mean_delta_rel,
                "group_mean_delta_rel_participants": group_mean_delta_rel_participants,
                "group_mean_delta_rel_abstainers": group_mean_delta_rel_abstainers,
                "group_mean_common_reward": group_mean_common_reward,
                "group_mean_personal_reward": group_mean_personal_reward,
                "group_mean_fee": group_mean_fee,
                "group_mean_altruism": group_mean_altruism,
                "group_mean_q_participation_participants": group_mean_q_participation_participants,
                "group_mean_q_participation_abstainers": group_mean_q_participation_abstainers,
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

        step = int(
            getattr(getattr(self.model, "scheduler", None), "steps", 0) or 0)
        agents = sorted(self.agents,
                        key=lambda a: int(getattr(a, "unique_id", 0)))
        eligible = [a for a in agents if
                    bool(getattr(a, "eligible_for_election", False))]
        participants = [a for a in eligible if
                        bool(getattr(a, "participating", False))]
        abstainers = [a for a in eligible if
                      not bool(getattr(a, "participating", False))]

        record = {
            "step": step,
            "area_id": getattr(self, "unique_id", None),
            "num_agents": len(agents),
            "num_eligible": len(eligible),
            "num_participants": len(participants),
            "num_abstainers": len(abstainers),
            "dist_to_reality": float(
                self._dist_to_reality) if self._dist_to_reality is not None else float(
                "nan"),
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
            "reward_threshold_common": float(
                getattr(self.model, "reward_threshold_common", float("nan"))),
            "reward_threshold_personal": float(
                getattr(self.model, "reward_threshold_personal", float("nan"))),
            "reward_rate_common": float(
                getattr(self.model, "reward_rate_common", float("nan"))),
            "reward_rate_personal": float(
                getattr(self.model, "reward_rate_personal", float("nan"))),
            "abstention_share": float(
                getattr(self.model, "abstention_share", float("nan"))),
            "agents": [self._snapshot_agent(a) for a in agents],
        }

        self._debug_history.append(record)
        max_steps = self._debug_max_steps()
        if len(self._debug_history) > max_steps:
            self._debug_history = self._debug_history[-max_steps:]

    def _capture_area_snapshot_for_logger(self) -> None:
        # --- optional schema v2 hook: snapshot right before mutation (pre-mutation)
        snapshot_sink = getattr(self.model, "_schema_v2_area_snapshot_sink", None)
        if snapshot_sink is not None:
            election_cost_rate = self.model.election_cost_rate
            fee_pool = self._election_fee_pool
            eligible_voters = self.num_agents
            participants = self.num_agents_participated_last
            turnout = self.voter_turnout
            dist_to_reality = self.dist_to_reality
            area_color = None
            if self.color_distribution is not None:
                area_color = self.color_distribution.copy()
            elected_color = None
            if self.voted_ordering is not None:
                elected_color = self.voted_ordering.copy()
            snapshot_sink(
                area=self,
                snapshot={
                    "election_cost_rate": election_cost_rate,
                    "fee_pool": fee_pool,
                    "eligible_voters": eligible_voters,
                    "participants": participants,
                    "turnout": turnout,
                    "dist_to_reality": dist_to_reality,
                    "area_color": area_color,
                    "elected_color": elected_color
                },
            )

    def mutate_cells(self) -> None:
        """Mutate cell colors based on the last election outcome."""
        if self.voter_turnout == 0:
            return
        # Take some number of cells to mutate (i.e., 5 %)
        n_to_mutate = int(self.model.mu * self.num_cells)
        # TODO/Idea: What if the voter_turnout determines the mutation rate?
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

    def step(self) -> None:
        """
        Run one step of the simulation.

        Conduct an election in the area,
        mutate the cells' colors according to the election outcome
        and update the color distribution of the area.
        """
        self.conduct_election()
        self.mutate_cells()
