from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
from mesa.visualization import TextElement

from src.viz.color_palette import get_group_color
from src.viz.factory import COLORS
from src.viz.helpers import (
    float_array,
    ordering_rank_matrix,
    safe_mean,
    safe_median,
    save_plot_to_base64,
    series_at,
    vector_matrix,
)


class MatplotlibElement(TextElement):
    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == 0:
            return ""
        data = model.datacollector.get_model_vars_dataframe()
        collective_assets = data.get("collective_assets") if data is not None else None
        if collective_assets is None:
            return ""
        fig, ax = plt.subplots()
        ax.plot(collective_assets, label="Collective assets")
        ax.set_title("Collective Assets Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Collective Assets")
        ax.legend()
        return save_plot_to_base64(fig)


class StepsTextElement(TextElement):
    def render(self, model) -> str:
        step = model.scheduler.steps
        first_agents = [str(a) for a in model.voting_agents[:5]]
        return (
            f"Step: {step} | cells: {len(model.color_cells)} | "
            f"areas: {len(model.areas)} | First 5 voters of "
            f"{len(model.voting_agents)}: {first_agents}"
        )


class AgentLearningHistograms(TextElement):
    """Fast feedback panel for live tuning. Not part of the main demo path."""

    def render(self, model) -> str:
        step = model.scheduler.steps
        agents = [a for a in model.voting_agents if a is not None]
        if not agents:
            return ""

        p = np.asarray([a.participation_probability() for a in agents], dtype=np.float64)
        q = np.asarray([a.q_participation for a in agents], dtype=np.float64)
        altruism = np.asarray([a.altruism_factor for a in agents], dtype=np.float64)
        assets = np.asarray([a.assets for a in agents], dtype=np.float64)

        if p.size == 0:
            return ""

        def finite(values: np.ndarray) -> np.ndarray:
            return values[np.isfinite(values)]

        p_f = finite(p)
        a_f = finite(altruism)
        if p_f.size == 0 or a_f.size == 0:
            return ""

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

        ax = axes[0][0]
        ax.hist(p_f, bins=20, range=(0.0, 1.0), color="gray", alpha=0.8)
        ax.set_title(f"p_participation (mean={np.mean(p_f):.3f}, med={np.median(p_f):.3f})")
        ax.set_xlim(0.0, 1.0)

        ax = axes[0][1]
        ax.hist(finite(q), bins=20, color="black", alpha=0.8)
        ax.set_title("q_participation")

        ax = axes[1][0]
        ax.hist(a_f, bins=20, range=(0.0, 1.0), color="blue", alpha=0.8)
        ax.set_title(f"altruism_factor (mean={np.mean(a_f):.3f}, med={np.median(a_f):.3f})")
        ax.set_xlim(0.0, 1.0)

        ax = axes[1][1]
        ax.hist(finite(assets), bins=20, color="green", alpha=0.8)
        ax.set_title("assets")

        fig.suptitle(f"Agent learning inspector (step={step})")
        plt.tight_layout()
        return save_plot_to_base64(fig)


class AreaDiagnosticsPanel(TextElement):
    """Per-area diagnostics panel for detailed debugging."""

    def __init__(self, max_steps: int = 10):
        super().__init__()
        self.max_steps = max_steps
        self._cached_step = -1
        self._cached_html = ""

    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == 0:
            return ""
        if step == self._cached_step:
            return self._cached_html

        areas = [a for a in model.areas if a is not None and a.unique_id != -1]
        if not areas:
            return ""
        areas = sorted(areas, key=lambda a: a.unique_id)

        data = model.datacollector.get_agent_vars_dataframe()
        if data is None or data.empty:
            return ""
        if not {"area_color_distribution", "quality_distance", "elected_color"}.issubset(data.columns):
            return ""

        recent = data.groupby(level=1, sort=False).tail(self.max_steps)
        area_frames = {area_id: frame for area_id, frame in recent.groupby(level=1, sort=False)}
        sample_frame = next(iter(area_frames.values()), None)
        if sample_frame is None or sample_frame.empty:
            return ""

        num_colors = len(sample_frame["area_color_distribution"].iloc[0])
        fig, axes = plt.subplots(
            nrows=len(areas) * 3,
            ncols=3,
            figsize=(14, 10.5 * len(areas)),
            sharex=False,  # type: ignore[arg-type]
        )

        for i, area in enumerate(areas):
            frame = area_frames.get(area.unique_id)
            if frame is None or frame.empty:
                continue

            row_top = i * 3
            row_mid = row_top + 1
            row_bot = row_top + 2
            steps = frame.index.get_level_values(0).to_numpy(dtype=float)
            color_matrix = vector_matrix(frame["area_color_distribution"].tolist(), num_colors)
            quality_series = float_array(frame["quality_distance"].tolist())
            elected_ranks = ordering_rank_matrix(frame["elected_color"].tolist(), num_colors)

            dist_series = None
            if "dist_to_reality" in frame.columns:
                dist_series = float_array(frame["dist_to_reality"].tolist())

            puzzle_distance_series = None
            if "puzzle_distance" in frame.columns:
                puzzle_distance_series = float_array(frame["puzzle_distance"].tolist())

            puzzle_color_matrix = None
            if "puzzle_color_distribution" in frame.columns and frame["puzzle_color_distribution"].notna().any():
                puzzle_color_matrix = vector_matrix(frame["puzzle_color_distribution"].tolist(), num_colors)

            ax0 = axes[row_top][0]
            ax1 = axes[row_top][1]
            ax2 = axes[row_top][2]

            ax0.plot(steps, quality_series, color="black", linestyle="--", linewidth=1.6, label="quality_distance")
            q_mode = model.quality_target_mode.strip().lower()
            if q_mode == "puzzle":
                if puzzle_distance_series is not None:
                    ax0.plot(steps, puzzle_distance_series, color="tab:blue", linestyle=":", linewidth=1.0, alpha=0.8, label="puzzle_distance")
                if dist_series is not None:
                    ax0.plot(steps, dist_series, color="tab:green", linestyle=":", linewidth=1.0, alpha=0.8, label="dist_to_reality")
            elif puzzle_distance_series is not None:
                ax0.plot(steps, puzzle_distance_series, color="tab:blue", linestyle=":", linewidth=1.0, alpha=0.8, label="puzzle_distance")

            for color_idx in range(num_colors):
                ax0.plot(steps, color_matrix[:, color_idx], color=COLORS[color_idx], linewidth=1.2, label=f"grid c{color_idx}")
                if puzzle_color_matrix is not None:
                    ax0.plot(steps, puzzle_color_matrix[:, color_idx], color=COLORS[color_idx], linestyle="--", linewidth=1.0, alpha=0.8, label=f"puzzle c{color_idx}")
            source = "puzzle_distance" if q_mode == "puzzle" else "dist_to_reality"
            ax0.set_title(f"Area {area.unique_id} grid/puzzle color-dst | quality_distance ({source})")
            ax0.set_xlabel("Step")
            ax0.set_ylabel("Color dist")
            ax0.legend(fontsize=6, loc="best")

            for color_id in range(num_colors):
                valid = elected_ranks[:, color_id] >= 0
                if np.any(valid):
                    ax1.plot(steps[valid], elected_ranks[valid, color_id], marker="o", label=f"Color {color_id}", color=COLORS[color_id], linewidth=0.2)
            ax1.set_title("Elected ordering")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Rank")
            ax1.invert_yaxis()

            hist = area.diag_history[-self.max_steps:]
            if hist:
                hist_len = len(hist)
                hist_steps = np.arange(step - hist_len + 1, step + 1)
                group_personal = [h.get("group_mean_personal_reward", []) for h in hist]
                group_fee = [h.get("group_mean_fee", []) for h in hist]
                num_groups = len(group_personal[0]) if group_personal and group_personal[0] is not None else 0

                for group_idx in range(num_groups):
                    color = get_group_color(group_idx)
                    ax2.plot(hist_steps, series_at(group_idx, group_personal), color=color, linestyle="--", label=f"g{group_idx} pers")
                    ax2.plot(hist_steps, series_at(group_idx, group_fee), color=color, linestyle=":", label=f"g{group_idx} fee")

                ax2.axhline(0.0, color="k", linewidth=0.5)
                if num_groups <= 10:
                    ax2.legend(fontsize=6)
            ax2.set_title("mean rewards/fees by group")
            ax2.set_xlabel("Step")

            ax3 = axes[row_mid][0]
            ax4 = axes[row_mid][1]
            ax5 = axes[row_mid][2]

            if hist:
                hist_len = len(hist)
                hist_steps = np.arange(step - hist_len + 1, step + 1)
                group_turnout = [h.get("group_turnout", []) for h in hist]
                overall_turnout = [h.get("turnout", float("nan")) for h in hist]
                group_assets = [h.get("group_mean_assets", []) for h in hist]
                group_delta_p = [h.get("group_mean_delta_rel_participants", []) for h in hist]
                group_delta_a = [h.get("group_mean_delta_rel_abstainers", []) for h in hist]
                num_groups = len(group_turnout[0]) if group_turnout and group_turnout[0] is not None else 0

                for group_idx in range(num_groups):
                    color = get_group_color(group_idx)
                    ax3.plot(hist_steps, series_at(group_idx, group_turnout), color=color, label=f"g{group_idx}")
                    ax4.plot(hist_steps, series_at(group_idx, group_assets), color=color, label=f"g{group_idx}")
                    ax5.plot(hist_steps, series_at(group_idx, group_delta_p), color=color, linestyle=":", label=f"g{group_idx} p")
                    ax5.plot(hist_steps, series_at(group_idx, group_delta_a), color=color, linestyle="--", label=f"g{group_idx} a")

                ax3.plot(hist_steps, overall_turnout, color="gray", linewidth=1.2, label="overall")
                ax3.set_ylabel("%")
                ax5.axhline(0.0, color="k", linewidth=0.5)

                if num_groups <= 10:
                    ax3.legend(fontsize=6)
                    ax4.legend(fontsize=6)
                    ax5.legend(fontsize=6)

            ax3.set_title("turnout by group")
            ax4.set_title("mean assets by group")
            ax5.set_title("mean delta_rel by group")

            ax6 = axes[row_bot][0]
            ax7 = axes[row_bot][1]
            ax8 = axes[row_bot][2]

            if hist:
                hist_len = len(hist)
                hist_steps = np.arange(step - hist_len + 1, step + 1)
                group_altruism = [h.get("group_mean_altruism", []) for h in hist]
                group_q_p = [h.get("group_mean_q_participation_participants", []) for h in hist]
                group_q_a = [h.get("group_mean_q_participation_abstainers", []) for h in hist]
                group_dissatisfaction = [h.get("group_mean_dissatisfaction", []) for h in hist]
                num_groups = len(group_altruism[0]) if group_altruism and group_altruism[0] is not None else 0

                for group_idx in range(num_groups):
                    color = get_group_color(group_idx)
                    ax6.plot(hist_steps, series_at(group_idx, group_altruism), color=color, label=f"g{group_idx}")
                    ax7.plot(hist_steps, series_at(group_idx, group_q_p), color=color, linestyle=":", label=f"g{group_idx} p")
                    ax7.plot(hist_steps, series_at(group_idx, group_q_a), color=color, linestyle="--", label=f"g{group_idx} a")
                    ax8.plot(hist_steps, series_at(group_idx, group_dissatisfaction), color=color, label=f"g{group_idx}")

                if num_groups <= 10:
                    ax6.legend(fontsize=6)
                    ax7.legend(fontsize=6)
                    ax8.legend(fontsize=6)

            ax6.set_title("mean altruism by group")
            ax7.set_title("mean q_participation by group")
            ax8.set_title("mean dissatisfaction by group")

        plt.tight_layout()
        self._cached_html = save_plot_to_base64(fig)
        self._cached_step = step
        return self._cached_html


class CohortElectionLearningDiagnostics(TextElement):
    """Cohort-stratified election learning diagnostics for live debugging."""

    def __init__(self, top_k: int = 8):
        super().__init__()
        self.top_k = top_k

    def _compute(self, model):
        agents = [a for a in model.voting_agents if a is not None]
        if not agents:
            return None

        rows = []
        for agent in agents:
            if not agent.eligible_for_election:
                continue

            try:
                p_part = agent.participation_probability()
            except ValueError:
                p_part = float("nan")

            rows.append(
                (
                    agent.personality_group_idx,
                    agent.participating,
                    agent.election_delta_abs,
                    agent._fee,
                    agent.reward_personal,
                    agent.altruism_factor,
                    p_part,
                )
            )

        if not rows:
            return None

        arr = np.asarray(rows, dtype=np.float64)
        gid = arr[:, 0].astype(np.int64)
        participated = arr[:, 1].astype(bool)
        delta = arr[:, 2]
        fee = arr[:, 3]
        personal = arr[:, 4]
        altruism = arr[:, 5]
        p_part = arr[:, 6]

        unique_g, counts = np.unique(gid, return_counts=True)
        order = np.argsort(counts)[::-1]
        unique_g = unique_g[order]

        show_other = unique_g.size > 10
        show_groups = unique_g[: min(self.top_k, unique_g.size)] if show_other else unique_g
        other_groups = set(unique_g[len(show_groups):].tolist()) if show_other else set()

        labels = [str(group_id) for group_id in show_groups]
        group_ids = [group_id for group_id in show_groups]
        if show_other:
            labels.append("other")
            group_ids.append(-1)

        out = {
            "labels": labels,
            "group_ids": group_ids,
            "eligible": [],
            "participants": [],
            "abstainers": [],
            "rate": [],
            "delta_p_mean": [],
            "delta_p_median": [],
            "delta_a_mean": [],
            "delta_a_median": [],
            "fee_mean": [],
            "personal_mean": [],
            "altruism_mean": [],
            "p_part_mean": [],
        }

        def add_bucket(mask: np.ndarray) -> None:
            part_mask = mask & participated
            abst_mask = mask & (~participated)
            eligible = int(np.sum(mask))
            participants = int(np.sum(part_mask))
            abstainers = int(np.sum(abst_mask))

            out["eligible"].append(eligible)  # type: ignore[arg-type]
            out["participants"].append(participants)  # type: ignore[arg-type]
            out["abstainers"].append(abstainers)  # type: ignore[arg-type]
            out["rate"].append(participants / eligible if eligible > 0 else float("nan"))  # type: ignore[arg-type]
            out["delta_p_mean"].append(safe_mean(delta[part_mask]))  # type: ignore[arg-type]
            out["delta_p_median"].append(safe_median(delta[part_mask]))  # type: ignore[arg-type]
            out["delta_a_mean"].append(safe_mean(delta[abst_mask]))  # type: ignore[arg-type]
            out["delta_a_median"].append(safe_median(delta[abst_mask]))  # type: ignore[arg-type]
            out["fee_mean"].append(safe_mean(fee[part_mask]))  # type: ignore[arg-type]
            out["personal_mean"].append(safe_mean(personal[mask]))  # type: ignore[arg-type]
            out["altruism_mean"].append(safe_mean(altruism[mask]))  # type: ignore[arg-type]
            out["p_part_mean"].append(safe_mean(p_part[mask]))  # type: ignore[arg-type]

        for group_id in show_groups:
            add_bucket(gid == group_id)
        if show_other:
            add_bucket(np.isin(gid, np.asarray(list(other_groups), dtype=np.int64)))

        for key, values in list(out.items()):
            if key == "labels":
                continue
            dtype = np.int64 if key == "group_ids" else np.float64
            out[key] = np.asarray(values, dtype=dtype)

        return out

    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == 0:
            return ""

        stats = self._compute(model)
        if stats is None:
            return ""

        labels = stats["labels"]
        group_ids = stats["group_ids"]
        x = np.arange(len(labels))
        group_colors = [get_group_color(group_id) for group_id in group_ids]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))

        ax = axes[0][0]
        ax.bar(x, np.nan_to_num(stats["rate"], nan=0.0), color=group_colors, alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Participation rate by preference group")
        ax.set_ylabel("participant_count / eligible")

        ax = axes[0][1]
        width = 0.4
        ax.bar(x - width / 2, np.nan_to_num(stats["delta_p_mean"], nan=0.0), width=width, label="participants", color=group_colors, alpha=0.8)
        ax.bar(x + width / 2, np.nan_to_num(stats["delta_a_mean"], nan=0.0), width=width, label="abstainers", color=group_colors, alpha=0.4)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Mean delta_assets by action (eligible only)")
        ax.set_ylabel("mean delta_assets")
        ax.legend()

        ax = axes[1][0]
        ax.bar(x - width, np.nan_to_num(stats["fee_mean"], nan=0.0), width=width, label="fee (participants)", color=group_colors, alpha=0.8)
        ax.bar(x, np.nan_to_num(stats["personal_mean"], nan=0.0), width=width, label="personal reward", color=group_colors, alpha=0.6)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Reward/Fee means")
        ax.set_ylabel("mean component")
        ax.legend(fontsize=6)

        ax = axes[1][1]
        ax.plot(x, np.nan_to_num(stats["altruism_mean"], nan=0.0), marker="o", label="mean altruism_factor", color="blue")
        ax.plot(x, np.nan_to_num(stats["p_part_mean"], nan=0.0), marker="o", label="mean p_participation", color="black")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Mean altruism_factor and p_participation")
        ax.legend(fontsize=6)

        elig = stats["eligible"]
        parts = stats["participants"]
        abst = stats["abstainers"]
        fig.suptitle(
            f"Cohort election learning diagnostics (step={step}) | eligible/part/abst per group: "
            + ", ".join(f"{labels[i]}:{elig[i]}/{parts[i]}/{abst[i]}" for i in range(len(labels)))
        )
        plt.tight_layout()
        return save_plot_to_base64(fig)

class AreaAgentDebugPanel(TextElement):
    """Step-wise per-agent debug panel for a single area.

    Renders a compact text overview of:
      - per-agent full state (including learning variables)
      - cast votes per agent
      - preference profile and aggregated ordering
      - reward components and deltas
    Intended for tiny configurations.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        area_id: int | None = None,
        max_agents: int | None = None,
        max_field_len: int | None = None,
        enabled: bool = True,
    ):
        super().__init__()
        self.max_steps = max(1, int(max_steps if max_steps is not None else 1))
        self.area_id = area_id
        self.max_agents = int(max_agents if max_agents is not None else 50)
        self.max_field_len = int(max_field_len if max_field_len is not None else 180)
        self.enabled = bool(enabled)
        self.max_list_items = 8
        self.agent_key_order = [
            "id",
            "pos",
            "assets",
            "assets_pre_est",
            "personality_group_idx",
            "personality_group",
            "personality",
            "est_real_dist",
            "confidence",
            "altruism_factor",
            "dissatisfaction_value",
            "dissatisfaction_baseline",
            "dissatisfaction_signal",
            "eligible",
            "participating",
            "q_participation",
            "p_participation",
            "participation_baseline",
            "participation_signal",
            "fee",
            "num_elections_participated",
            "reward_personal",
            "delta_abs",
            "delta_rel",
            "known_cells_count",
            "known_cells",
            "participation_strategy",
            "voting_strategy",
            "award_history_tail",
        ]

    def _truncate(self, s: str) -> str:
        if len(s) <= self.max_field_len:
            return s
        return s[: max(0, self.max_field_len - 3)] + "..."

    def _fmt_value(self, value) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.6g}"
        if isinstance(value, (int, bool)):
            return str(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            items = []
            for v in list(value)[: self.max_list_items]:
                items.append(self._fmt_value(v))
            s = "[" + ", ".join(items)
            if len(value) > self.max_list_items:
                s += ", ..."
            s += "]"
            return self._truncate(s)
        if isinstance(value, dict):
            items = []
            for k in list(value.keys())[: self.max_list_items]:
                items.append(f"{k}={self._fmt_value(value[k])}")
            s = "{" + ", ".join(items)
            if len(value) > self.max_list_items:
                s += ", ..."
            s += "}"
            return self._truncate(s)
        return self._truncate(str(value))

    def _format_record(self, rec: dict) -> str:
        lines: list[str] = []
        lines.append(
            f"Agent Debug | area {rec.get('area_id')} | step {rec.get('step')}"
        )
        lines.append(
            "Summary: "
            f"agents={rec.get('num_agents')} "
            f"eligible={rec.get('num_eligible')} "
            f"participants={rec.get('num_participants')} "
            f"abstainers={rec.get('num_abstainers')}"
        )
        lines.append(
            "Outcome: "
            f"quality_distance={self._fmt_value(rec.get('quality_distance'))} "
            f"(source={rec.get('quality_distance_source')}) "
            f"winning_option={rec.get('winning_option')} "
            f"aggregated_ordering={self._fmt_value(rec.get('aggregated_ordering'))}"
        )
        lines.append(
            "Distances: "
            f"dist_to_reality={self._fmt_value(rec.get('dist_to_reality'))} "
            f"puzzle_distance={self._fmt_value(rec.get('puzzle_distance'))}"
        )
        if rec.get("voted_ordering") is not None:
            lines.append(f"Voted ordering: {self._fmt_value(rec.get('voted_ordering'))}")
        if rec.get("real_color_distribution") is not None:
            lines.append(f"Actual color distribution: {self._fmt_value(rec.get('real_color_distribution'))}")
        lines.append(
            "Rewards: "
            f"quality_threshold={self._fmt_value(rec.get('quality_threshold_common'))} "
            f"reward_rate={self._fmt_value(rec.get('reward_rate'))}"
        )

        #pref = rec.get("preference_profile") or []
        #rows = len(pref)
        #cols = len(pref[0]) if rows > 0 else 0
        #lines.append(f"Preference profile shape: {rows} x {cols}")
        lines.append("Votes:")
        votes = rec.get("votes") or []
        if not votes:
            lines.append("  (no votes recorded)")
        else:
            for vote in votes:
                lines.append(
                    f"  Agent {vote.get('agent_id')}: "
                    f"scores={self._fmt_value(vote.get('scores'))} "
                    f"ordering (debug-tie-break!): "
                    f"{self._fmt_value(vote.get('ordering'))}"
                )

        lines.append("Agents:")
        agents = rec.get("agents") or []
        shown_agents = agents[: self.max_agents]
        for agent in shown_agents:
            agent_id = agent.get("id")
            lines.append(f"  Agent {agent_id}:")
            pair_of_entries = ""  # Save some lines by combining two fields
            for cnt, key in enumerate(self.agent_key_order):
                if key in agent:
                    pair_of_entries += f"{key}: {self._fmt_value(agent.get(key))}  "
                    if (cnt + 1) % 2 == 0:
                        lines.append(f"    {pair_of_entries}")
                        pair_of_entries = ""
            if len(pair_of_entries) > 0:  # In case of odd number of fields
                lines.append(f"    {pair_of_entries}")
        if len(agents) > self.max_agents:
            lines.append(f"  ... ({len(agents) - self.max_agents} more agents omitted)")

        return "\n".join(lines)

    def render(self, model) -> str:
        if not self.enabled:
            return ""

        # Enable capture for subsequent steps.
        setattr(model, "_debug_agent_panel_enabled", True)
        setattr(model, "_debug_agent_panel_max_steps", int(self.max_steps))

        step = int(model.scheduler.steps)
        if step == 0:
            return ""

        areas = [a for a in model.areas if a is not None and a.unique_id != -1]
        if not areas:
            return ""
        areas = sorted(areas, key=lambda a: int(a.unique_id))

        area = None
        if self.area_id is None:
            area = areas[0]
        else:
            for a in areas:
                if int(a.unique_id) == int(self.area_id):
                    area = a
                    break
        if area is None:
            return ""

        history = getattr(area, "debug_history", []) or []
        if not history:
            return "<pre>Agent debug capture not yet available (advance one step).</pre>"

        records = history[-self.max_steps :]
        text_blocks = [self._format_record(rec) for rec in records]
        text = ("\n\n" + ("-" * 60) + "\n\n").join(text_blocks)
        return f"<pre>{html.escape(text)}</pre>"
