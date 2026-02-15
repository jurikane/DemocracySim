from __future__ import annotations

import matplotlib.pyplot as plt
from mesa.visualization import TextElement
import matplotlib.patches as patches
from src.viz.factory import COLORS, get_vis_cfg
import base64
import math
import io
import numpy as np


vis_cfg = get_vis_cfg()
show_area_stats = bool(getattr(vis_cfg, 'show_area_stats', True))


def _series_at(idx: int, seqs: list) -> list[float]:
    return [s[idx] if s is not None and len(s) > idx else float("nan") for s in seqs]


def save_plot_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}" />'


class AreaDiagnosticsPanel(TextElement):
    """Per-area diagnostics panel.

    Plots last N steps for each area with three rows of three columns:
      Row 1 (AreaStats):
        1) area color distribution + dist_to_reality
        2) elected ordering
        3) mean common + mean personal rewards
      Row 2 (Diagnostics):
        4) turnout by personality_group
        5) mean assets by personality_group
        6) mean delta_rel by personality_group
      Row 3 (Reserved):
        7) mean altruism by personality_group
        8) mean q_participation by personality_group (participants dotted)
        9) mean dissatisfaction by personality_group
    """

    def __init__(self, max_steps: int = 10):
        super().__init__()
        self.max_steps = int(max_steps)

    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == 0:
            return ""

        areas = [a for a in model.areas if a is not None and a.unique_id != -1]
        if not areas:
            return ""
        areas = sorted(areas, key=lambda a: int(a.unique_id))

        # Diagnostics histories
        histories = []
        for area in areas:
            hist = getattr(area, "diag_history", [])
            histories.append(hist[-self.max_steps:] if hist else [])

        # AreaStats series from datacollector
        data = model.datacollector.get_agent_vars_dataframe()
        if data is None or len(data) == 0:
            return ""
        if ('area_color_distribution' not in data.columns
                or 'dist_to_reality' not in data.columns
                or 'elected_color' not in data.columns):
            return ""

        color_distribution = data['area_color_distribution'].dropna()
        dist_to_reality = data['dist_to_reality'].dropna()
        election_results = data['elected_color'].dropna()

        if len(color_distribution) == 0:
            return ""

        num_colors = len(color_distribution.iloc[0])
        num_areas = len(areas)
        fig, axes = plt.subplots(
            nrows=num_areas * 3,
            ncols=3,
            figsize=(14, 10.5 * num_areas),
            sharex=False,  # type: ignore[arg-type]
        )

        for i, area in enumerate(areas):
            row_top = i * 3
            row_mid = row_top + 1
            row_bot = row_top + 2

            # --- AreaStats (top row) ---
            area_cd = color_distribution.xs(area.unique_id, level=1)
            area_dist = dist_to_reality.xs(area.unique_id, level=1)
            area_elec = election_results.xs(area.unique_id, level=1)

            # limit to last N steps for AreaStats
            area_cd = area_cd.tail(self.max_steps)
            area_dist = area_dist.tail(self.max_steps)
            area_elec = area_elec.tail(self.max_steps)

            ax0 = axes[row_top][0]
            ax1 = axes[row_top][1]
            ax2 = axes[row_top][2]

            ax0.plot(area_dist.index, area_dist.values, color='Black', linestyle='--')
            for color_idx in range(num_colors):
                cdata = area_cd.apply(lambda x: x[color_idx])
                ax0.plot(cdata.index, cdata.values, color=COLORS[color_idx])
            ax0.set_title(f'Area {area.unique_id} color-dst | --- dist_to_reality')
            ax0.set_xlabel('Step')
            ax0.set_ylabel('Color dist')

            for color_id in range(num_colors):
                cdata = area_elec.apply(lambda x: list(x).index(color_id) if color_id in x else None)
                ax1.plot(cdata.index, cdata.values, marker='o',
                         label=f'Color {color_id}', color=COLORS[color_id],
                         linewidth=0.2)
            ax1.set_title('Elected ordering')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Rank')
            ax1.invert_yaxis()

            # rewards (top row, col 3)
            hist = histories[i]
            if hist:
                hist_len = len(hist)
                step_axis = np.arange(int(step) - hist_len + 1, int(step) + 1)
                group_common = [h.get("group_mean_common_reward", []) for h in hist]
                group_personal = [h.get("group_mean_personal_reward", []) for h in hist]
                group_fee = [h.get("group_mean_fee", []) for h in hist]

                num_groups = len(group_common[0]) if group_common and group_common[0] is not None else 0
                cmap = plt.get_cmap("tab10")

                for g in range(num_groups):
                    c_series = _series_at(g, group_common)
                    p_series = _series_at(g, group_personal)
                    f_series = _series_at(g, group_fee)
                    color = cmap(g % 10)

                    ax2.plot(step_axis, c_series, color=color, label=f"g{g} com")
                    ax2.plot(step_axis, p_series, color=color, linestyle="--", label=f"g{g} pers")
                    ax2.plot(step_axis, f_series, color=color, linestyle=":", label=f"g{g} fee")

                ax2.axhline(0.0, color="k", linewidth=0.5)
                if num_groups <= 10:
                    ax2.legend(fontsize=6)
            ax2.set_title("mean rewards/fees by group")
            ax2.set_xlabel("Step")
            #ax2.legend(fontsize=6)

            # --- Diagnostics (middle row) ---
            ax3 = axes[row_mid][0]
            ax4 = axes[row_mid][1]
            ax5 = axes[row_mid][2]

            if hist:
                hist_len = len(hist)
                step_axis = np.arange(int(step) - hist_len + 1, int(step) + 1)

                group_turnout = [h.get("group_turnout", []) for h in hist]
                overall_turnout = [h.get("turnout", float("nan")) for h in hist]
                group_assets = [h.get("group_mean_assets", []) for h in hist]
                group_delta_p = [h.get("group_mean_delta_rel_participants", []) for h in hist]
                group_delta_a = [h.get("group_mean_delta_rel_abstainers", []) for h in hist]

                num_groups = len(group_turnout[0]) if group_turnout and group_turnout[0] is not None else 0
                cmap = plt.get_cmap("tab10")

                for g in range(num_groups):
                    t_series = _series_at(g, group_turnout)
                    a_series = _series_at(g, group_assets)
                    dp_series = _series_at(g, group_delta_p)
                    da_series = _series_at(g, group_delta_a)
                    color = cmap(g % 10)

                    ax3.plot(step_axis, t_series, color=color, label=f"g{g}")
                    ax4.plot(step_axis, a_series, color=color, label=f"g{g}")
                    ax5.plot(step_axis, dp_series, color=color, linestyle=":", label=f"g{g} p")
                    ax5.plot(step_axis, da_series, color=color, label=f"g{g} a")

                ax3.plot(step_axis, overall_turnout, color="gray", linewidth=1.2, label="overall")
                ax3.set_ylabel("%")
                ax5.axhline(0.0, color="k", linewidth=0.5)

                if num_groups <= 10:
                    ax3.legend(fontsize=6)
                    ax4.legend(fontsize=6)
                    ax5.legend(fontsize=6)

            ax3.set_title("turnout by group")
            ax4.set_title("mean assets by group")
            ax5.set_title("mean delta_rel by group")

            # --- Reserved (bottom row) ---
            ax6 = axes[row_bot][0]
            ax7 = axes[row_bot][1]
            ax8 = axes[row_bot][2]

            if hist:
                hist_len = len(hist)
                step_axis = np.arange(int(step) - hist_len + 1, int(step) + 1)

                group_altruism = [h.get("group_mean_altruism", []) for h in hist]
                group_q_p = [h.get("group_mean_q_participation_participants", []) for h in hist]
                group_q_a = [h.get("group_mean_q_participation_abstainers", []) for h in hist]
                group_dissatisfaction = [h.get("group_mean_dissatisfaction", []) for h in hist]

                num_groups = len(group_altruism[0]) if group_altruism and group_altruism[0] is not None else 0
                cmap = plt.get_cmap("tab10")

                for g in range(num_groups):
                    a_series = _series_at(g, group_altruism)
                    qp_series = _series_at(g, group_q_p)
                    qa_series = _series_at(g, group_q_a)
                    s_series = _series_at(g, group_dissatisfaction)
                    color = cmap(g % 10)

                    ax6.plot(step_axis, a_series, color=color, label=f"g{g}")
                    ax7.plot(step_axis, qp_series, color=color, linestyle=":", label=f"g{g} p")
                    ax7.plot(step_axis, qa_series, color=color, label=f"g{g} a")
                    ax8.plot(step_axis, s_series, color=color, label=f"g{g}")

                if num_groups <= 10:
                    ax6.legend(fontsize=6)
                    ax7.legend(fontsize=6)
                    ax8.legend(fontsize=6)

            ax6.set_title("mean altruism by group")
            ax7.set_title("mean q_participation by group")
            ax8.set_title("mean dissatisfaction by group")

        plt.tight_layout()
        return save_plot_to_base64(fig)


class PersonalityGroupDistribution(TextElement):
    def __init__(self):
        super().__init__()
        self.pers_dist_plot = None

    def create_once(self, model):
        dists = model.personality_group_distribution
        personality_groups = model.personality_groups
        num_personality_groups = personality_groups.shape[0]
        num_agents = model.num_agents
        colors = COLORS[:model.num_colors]
        num_colors = len(personality_groups[0])

        fig, ax = plt.subplots(figsize=(6, 4))
        heights = dists
        bars = ax.bar(range(num_personality_groups), heights, width=0.6)

        for bar, personality_group in zip(bars, personality_groups):
            height = bar.get_height()
            width = bar.get_width()
            for i, color_idx in enumerate(personality_group):
                rect_width = width / num_colors
                coords = (bar.get_x() + i * rect_width, 0)
                rect = patches.Rectangle(coords, rect_width, height,
                                         color=colors[color_idx])
                ax.add_patch(rect)

        ax.set_xlabel('"Personality Group" ID')
        ax.set_ylabel(f'Percentage of the {num_agents} Agents')
        ax.set_title('Global distribution of personality groups among agents')
        plt.tight_layout()
        self.pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model) -> str:
        if model.scheduler.steps == 0:
            self.create_once(model)
        return self.pers_dist_plot or ""


class _AreaTimeSeriesElement(TextElement):
    """Base class for per-area time series plots backed by agent vars dataframe."""

    series_column: str = ""
    title: str = ""
    ylabel: str = ""

    def _get_series(self, model):
        data = model.datacollector.get_agent_vars_dataframe()
        if data is None or data.empty:
            return None
        if self.series_column not in data.columns:
            return None
        series = data[self.series_column].dropna()
        return None if series.empty else series

    @staticmethod
    def _line_style(i: int) -> str:
        if i < 10:
            return "-"
        if i < 20:
            return ":"
        return "--"

    def render(self, model) -> str:
        series = self._get_series(model)
        if series is None:
            return ""

        area_ids = series.index.get_level_values(1).unique()
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, area_id in enumerate(area_ids):
            # If index isn't a MultiIndex with that level, let it fail loudly.
            area_data = series.xs(area_id, level=1)
            ax.plot(
                area_data.index,
                area_data.values,
                label=f"Area {area_id}",
                linestyle=self._line_style(i),
            )

        ax.set_title(self.title)
        ax.set_xlabel("Step")
        ax.set_ylabel(self.ylabel)
        ax.legend()
        return save_plot_to_base64(fig)


class VoterTurnoutElement(_AreaTimeSeriesElement):
    series_column = "turnout"
    title = "Voter Turnout by Area Over Time"
    ylabel = "Voter Turnout (%)"


class AreaGiniElement(_AreaTimeSeriesElement):
    series_column = "gini_index"
    title = "Gini Index by Area Over Time"
    ylabel = "Gini Index (0-100)"

class MatplotlibElement(TextElement):
    def render(self, model) -> str:
        step = model.scheduler.steps
        if not show_area_stats or step == 0:
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
        return (f"Step: {step} | cells: {len(model.color_cells)} | "
                f"areas: {len(model.areas)} | First 5 voters of "
                f"{len(model.voting_agents)}: {first_agents}")


class ReplayGridStepStatusElement(TextElement):
    """Show replay step/grid alignment status (useful for sparse grid snapshots)."""

    def render(self, model) -> str:
        if not hasattr(model, "replay_recorded_step") or not hasattr(model, "replay_grid_source_step"):
            return ""

        rec = int(getattr(model, "replay_recorded_step", getattr(model.scheduler, "steps", 0)))
        src = int(getattr(model, "replay_grid_source_step", rec))

        if rec == src:
            return (
                f"<div style='padding:6px 8px; margin:4px 0; border:1px solid #d7d7d7; "
                f"background:#f8f8f8; font-family:monospace;'>"
                f"Replay Step {rec} | Grid Step {src}"
                f"</div>"
            )

        return (
            f"<div style='padding:6px 8px; margin:4px 0; border:1px solid #d5a200; "
            f"background:#fff8dc; font-family:monospace;'>"
            f"Replay Step {rec} | Grid Step {src} (carry-forward snapshot)"
            f"</div>"
        )


class AreaPersonalityGroupDists(TextElement):
    def __init__(self):
        super().__init__()
        self.areas_pers_dist_plot = None

    def create_once(self, model):
        colors = COLORS[:model.num_colors]
        personality_groups = model.personality_groups
        num_colors = len(personality_groups[0])
        num_personality_groups = personality_groups.shape[0]
        num_areas = len(model.areas)

        if num_areas == 0:
            self.areas_pers_dist_plot = ""
            return

        num_cols = math.ceil(math.sqrt(num_areas))
        num_rows = math.ceil(num_areas / num_cols)
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(8, num_areas),
            sharex=True,  # type: ignore[arg-type]
        )
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, area in zip(axes_flat, model.areas):
            p_dist = area.personality_group_distribution
            num_agents = area.num_agents
            heights = [int(val * num_agents) for val in p_dist] if p_dist else []
            bars = ax.bar(range(num_personality_groups), heights, color='skyblue')
            max_height = max(heights) if heights else 1
            p_top_height = max_height * 0.02

            for bar, personality_group in zip(bars, personality_groups):
                height = bar.get_height()
                width = bar.get_width()
                for i, color_idx in enumerate(personality_group):
                    rect_width = width / num_colors
                    coords = (bar.get_x() + i * rect_width, height)
                    rect = patches.Rectangle(coords, rect_width, p_top_height,
                                             color=colors[color_idx])
                    ax.add_patch(rect)

            ax.set_xlabel('"Personality Group" ID')
            ax.set_ylabel('Number of Agents')
            ax.set_title(f'Area {area.unique_id}')

        plt.tight_layout()
        self.areas_pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model) -> str:
        if model.scheduler.steps == 0:
            self.create_once(model)
        return self.areas_pers_dist_plot or ""


class AgentLearningHistograms(TextElement):
    """Fast feedback panel: per-step histograms over voting agents.

    Reads agent state directly from model.voting_agents (no schema/logging changes).
    Renders 4 histograms:
      - participation probability p
      - q_participation
      - altruism_factor
      - assets
    and prints mean/median for p and altruism.

    Intended for live sanity checks while tuning knobs.
    """

    def render(self, model) -> str:
        step = int(model.scheduler.steps)
        agents = [a for a in model.voting_agents if a is not None]
        if not agents:
            return ""

        # Collect vectors (fail loudly if model/agent contract is broken)
        p = np.asarray([float(a.participation_probability()) for a in agents], dtype=np.float64)
        q = np.asarray([float(a.q_participation) for a in agents], dtype=np.float64)
        altruism = np.asarray([float(a.altruism_factor) for a in agents], dtype=np.float64)
        assets = np.asarray([float(a.assets) for a in agents], dtype=np.float64)

        # Require something meaningful
        if p.size == 0:
            return ""

        # Summary stats (finite only)
        def _finite(x: np.ndarray) -> np.ndarray:
            return x[np.isfinite(x)]

        p_f = _finite(p)
        a_f = _finite(altruism)
        if p_f.size == 0 or a_f.size == 0:
            return ""

        p_mean = float(np.mean(p_f))
        p_median = float(np.median(p_f))
        a_mean = float(np.mean(a_f))
        a_median = float(np.median(a_f))

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        ax = axes[0][0]
        ax.hist(p_f, bins=20, range=(0.0, 1.0), color="gray", alpha=0.8)
        ax.set_title(f"p_participation (mean={p_mean:.3f}, med={p_median:.3f})")
        ax.set_xlim(0.0, 1.0)

        ax = axes[0][1]
        q_f = _finite(q)
        ax.hist(q_f, bins=20, color="black", alpha=0.8)
        ax.set_title("q_participation")

        ax = axes[1][0]
        ax.hist(a_f, bins=20, range=(0.0, 1.0), color="blue", alpha=0.8)
        ax.set_title(f"altruism_factor (mean={a_mean:.3f}, med={a_median:.3f})")
        ax.set_xlim(0.0, 1.0)

        ax = axes[1][1]
        assets_f = _finite(assets)
        ax.hist(assets_f, bins=20, color="green", alpha=0.8)
        ax.set_title("assets")

        fig.suptitle(f"Agent learning inspector (step={step})")
        plt.tight_layout()
        return save_plot_to_base64(fig)


class CohortElectionLearningDiagnostics(TextElement):
    """Cohort-stratified election learning diagnostics (live runs).

    Reads directly from model.voting_agents and uses per-election variables stored on agents.

    Per personality_group (cohort), we compute (eligible agents only):
      - counts + participation_rate
      - mean/median delta for participants vs abstainers
      - mean fee (participants), mean common reward, mean personal reward
      - optional: mean altruism_factor, mean participation_probability

    Plot layout:
      A: participation rate by group
      B: mean delta participants vs abstainers by group
      C: fee/common/personal means by group
      D: optional altruism mean by group

    If groups > 10: show top-K by population + 'other'.
    """

    def __init__(self, top_k: int = 8):
        super().__init__()
        self.top_k = int(top_k)

    @staticmethod
    def _safe_mean(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else float("nan")

    @staticmethod
    def _safe_median(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    def _compute(self, model):
        agents = [a for a in model.voting_agents if a is not None]
        if not agents:
            return None

        # Build per-agent rows (eligible only)
        rows = []
        for a in agents:
            eligible = bool(getattr(a, "eligible_for_election", False))  # Has to be robust to account for replay agent-stubs
            if not eligible:
                continue
            gid_i = int(a.personality_group_idx)

            participated = a.participating
            delta = a.election_delta_abs
            fee = getattr(a, "_fee")
            common = getattr(a, "_reward_common_comp")
            personal = getattr(a, "_reward_pers_comp")
            altruism = float(a.altruism_factor)
            try:
                p_part = float(a.participation_probability())
            except ValueError:
                p_part = float("nan")

            rows.append(
                (gid_i, participated, delta, fee, common, personal, altruism, p_part)
            )

        if not rows:
            return None

        arr = np.asarray(rows, dtype=np.float64)
        gid = arr[:, 0].astype(np.int64)
        participated = arr[:, 1].astype(bool)
        delta = arr[:, 2]
        fee = arr[:, 3]
        common = arr[:, 4]
        personal = arr[:, 5]
        altruism = arr[:, 6]
        p_part = arr[:, 7]

        # Determine groups to show
        unique_g, counts = np.unique(gid, return_counts=True)
        order = np.argsort(counts)[::-1]
        unique_g = unique_g[order]
        counts = counts[order]

        show_other = unique_g.size > 10
        if show_other:
            k = min(self.top_k, unique_g.size)
            show_groups = unique_g[:k]
            other_groups = set(unique_g[k:].tolist())
        else:
            show_groups = unique_g
            other_groups = set()

        labels = [str(int(g)) for g in show_groups]
        if show_other:
            labels.append("other")

        def _mask_for_group(gval):
            return gid == gval

        # Aggregate per shown group
        out = {
            "labels": labels,
            "eligible": [],
            "participants": [],
            "abstainers": [],
            "rate": [],
            "delta_p_mean": [],
            "delta_p_median": [],
            "delta_a_mean": [],
            "delta_a_median": [],
            "fee_mean": [],
            "common_mean": [],
            "personal_mean": [],
            "altruism_mean": [],
            "p_part_mean": [],
        }

        def _add_bucket(mask: np.ndarray):
            elig_n = int(np.sum(mask))
            part_mask = mask & participated
            abst_mask = mask & (~participated)
            part_n = int(np.sum(part_mask))
            abst_n = int(np.sum(abst_mask))
            rate = float(part_n / elig_n) if elig_n > 0 else float("nan")

            out["eligible"].append(elig_n)  # type: ignore[arg-type]
            out["participants"].append(part_n)  # type: ignore[arg-type]
            out["abstainers"].append(abst_n)  # type: ignore[arg-type]
            out["rate"].append(rate)  # type: ignore[arg-type]

            out["delta_p_mean"].append(self._safe_mean(delta[part_mask]))  # type: ignore[arg-type]
            out["delta_p_median"].append(self._safe_median(delta[part_mask]))  # type: ignore[arg-type]
            out["delta_a_mean"].append(self._safe_mean(delta[abst_mask]))  # type: ignore[arg-type]
            out["delta_a_median"].append(self._safe_median(delta[abst_mask]))  # type: ignore[arg-type]

            # Fee meaningful only for participants
            out["fee_mean"].append(self._safe_mean(fee[part_mask]))  # type: ignore[arg-type]
            out["common_mean"].append(self._safe_mean(common[mask]))  # type: ignore[arg-type]
            out["personal_mean"].append(self._safe_mean(personal[mask]))  # type: ignore[arg-type]
            out["altruism_mean"].append(self._safe_mean(altruism[mask]))  # type: ignore[arg-type]
            out["p_part_mean"].append(self._safe_mean(p_part[mask]))  # type: ignore[arg-type]

        for g in show_groups:
            _add_bucket(_mask_for_group(g))
        if show_other:
            other_mask = np.isin(gid, np.array(list(other_groups), dtype=np.int64))
            _add_bucket(other_mask)

        # Convert lists -> arrays
        for k in list(out.keys()):
            if k == "labels":
                continue
            out[k] = np.asarray(out[k], dtype=np.float64)

        return out

    def render(self, model) -> str:
        step = int(model.scheduler.steps)
        if step == 0:
            # Avoid noisy empty plots before the first election has happened.
            return ""

        stats = self._compute(model)
        if stats is None:
            return ""

        labels = stats["labels"]
        n = len(labels)
        x = np.arange(n)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))

        # Panel A: participation rate
        ax = axes[0][0]
        rate = stats["rate"]
        ax.bar(x, np.nan_to_num(rate, nan=0.0), color="gray", alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Participation rate by personality_group")
        ax.set_ylabel("participant_count / eligible")

        # Panel B: mean delta participants vs abstainers
        ax = axes[0][1]
        w = 0.4
        dp = stats["delta_p_mean"]
        da = stats["delta_a_mean"]
        ax.bar(x - w / 2, np.nan_to_num(dp, nan=0.0), width=w, label="participants", color="black", alpha=0.8)
        ax.bar(x + w / 2, np.nan_to_num(da, nan=0.0), width=w, label="abstainers", color="red", alpha=0.6)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Mean delta_assets by action (eligible only)")
        ax.set_ylabel("mean delta_assets")
        ax.legend()

        # Panel C: decomposition
        ax = axes[1][0]
        fee_m = stats["fee_mean"]
        common_m = stats["common_mean"]
        pers_m = stats["personal_mean"]
        ax.bar(x - w, np.nan_to_num(fee_m, nan=0.0), width=w, label="fee (participants)", color="orange", alpha=0.8)
        ax.bar(x, np.nan_to_num(common_m, nan=0.0), width=w, label="common reward", color="blue", alpha=0.6)
        ax.bar(x + w, np.nan_to_num(pers_m, nan=0.0), width=w, label="personal reward", color="green", alpha=0.6)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Reward decomposition means")
        ax.set_ylabel("mean component")
        ax.legend(fontsize=6)

        # Panel D: altruism + p
        ax = axes[1][1]
        altru = stats["altruism_mean"]
        pmean = stats["p_part_mean"]
        ax.plot(x, np.nan_to_num(altru, nan=0.0), marker="o", label="mean altruism_factor", color="blue")
        ax.plot(x, np.nan_to_num(pmean, nan=0.0), marker="o", label="mean p_participation", color="black")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Mean altruism_factor and p_participation")
        ax.legend(fontsize=6)

        elig = stats["eligible"].astype(int)
        parts = stats["participants"].astype(int)
        abst = stats["abstainers"].astype(int)
        fig.suptitle(
            f"Cohort election learning diagnostics (step={step}) | eligible/part/abst per group: "
            + ", ".join([f"{labels[i]}:{elig[i]}/{parts[i]}/{abst[i]}" for i in range(n)])
        )
        plt.tight_layout()
        return save_plot_to_base64(fig)
    
