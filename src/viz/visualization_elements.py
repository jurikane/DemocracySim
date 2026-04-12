from __future__ import annotations

import matplotlib.pyplot as plt
from mesa.visualization import TextElement
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from src.viz.factory import COLORS
from src.viz.color_palette import get_group_color
import base64
import math
import io
import numpy as np

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


def _float_array(values) -> np.ndarray:
    arr = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        arr[i] = np.nan if value is None else value
    return arr


def _vector_matrix(values, width: int) -> np.ndarray:
    matrix = np.full((len(values), width), np.nan, dtype=np.float64)
    for row_idx, value in enumerate(values):
        if value is None:
            continue
        row = np.asarray(value, dtype=np.float64)
        matrix[row_idx, : min(width, row.size)] = row[:width]
    return matrix


def _ordering_rank_matrix(values, width: int) -> np.ndarray:
    ranks = np.full((len(values), width), -1, dtype=np.int16)
    for row_idx, value in enumerate(values):
        if value is None:
            continue
        ordering = np.asarray(value, dtype=np.int16)
        limit = min(width, ordering.size)
        ranks[row_idx, ordering[:limit]] = np.arange(limit, dtype=np.int16)
    return ranks


def wrap_html_panel(
    content: str,
    *,
    title: str,
    collapsible: bool = True,
    default_open: bool = False,
) -> str:
    if not content:
        return ""
    if not collapsible:
        return content
    open_attr = " open" if default_open else ""
    return (
        f"<details{open_attr} style='margin:8px 0;'>"
        f"<summary style='cursor:pointer; font-weight:600;'>{title}</summary>"
        f"<div style='padding-top:8px'>{content}</div>"
        f"</details>"
    )


def _with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def _step_axis(model, model_vars: dict[str, list[object]], num_steps: int) -> np.ndarray:
    if "step" in model_vars:
        return _float_array(model_vars["step"])

    starts_at_zero = num_steps == model.scheduler.steps + 1
    start = 0.0 if starts_at_zero else 1.0
    return np.arange(start, start + num_steps, dtype=float)


class AreaDiagnosticsPanel(TextElement):
    """Per-area diagnostics panel.

    Plots last N steps for each area with three rows of three columns:
      Row 1 (AreaStats):
        1) area color distribution + quality_distance (mode-aware)
        2) elected ordering
        3) mean common + mean personal rewards
      Row 2 (Diagnostics):
        4) turnout by preference group
        5) mean assets by preference group
        6) mean delta_rel by preference group
      Row 3 (Reserved):
        7) mean altruism by preference group
        8) mean q_participation by preference group (participants dotted)
        9) mean dissatisfaction by preference group
    """

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
        required_columns = {
            "area_color_distribution",
            "quality_distance",
            "elected_color",
        }
        if not required_columns.issubset(data.columns):
            return ""

        recent = data.groupby(level=1, sort=False).tail(self.max_steps)
        area_frames = {
            area_id: frame
            for area_id, frame in recent.groupby(level=1, sort=False)
        }
        sample_frame = next(iter(area_frames.values()), None)
        if sample_frame is None or sample_frame.empty:
            return ""

        sample_colors = sample_frame["area_color_distribution"].iloc[0]
        num_colors = len(sample_colors)
        num_areas = len(areas)
        fig, axes = plt.subplots(
            nrows=num_areas * 3,
            ncols=3,
            figsize=(14, 10.5 * num_areas),
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
            color_matrix = _vector_matrix(frame["area_color_distribution"].tolist(), num_colors)
            quality_series = _float_array(frame["quality_distance"].tolist())
            elected_ranks = _ordering_rank_matrix(frame["elected_color"].tolist(), num_colors)

            dist_series = None
            if "dist_to_reality" in frame.columns:
                dist_series = _float_array(frame["dist_to_reality"].tolist())

            puzzle_distance_series = None
            if "puzzle_distance" in frame.columns:
                puzzle_distance_series = _float_array(frame["puzzle_distance"].tolist())

            puzzle_color_matrix = None
            if "puzzle_color_distribution" in frame.columns and frame["puzzle_color_distribution"].notna().any():
                puzzle_color_matrix = _vector_matrix(frame["puzzle_color_distribution"].tolist(), num_colors)

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
                    ax0.plot(
                        steps,
                        puzzle_color_matrix[:, color_idx],
                        color=COLORS[color_idx],
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                        label=f"puzzle c{color_idx}",
                    )
            source = "puzzle_distance" if q_mode == "puzzle" else "dist_to_reality"
            ax0.set_title(f"Area {area.unique_id} grid/puzzle color-dst | quality_distance ({source})")
            ax0.set_xlabel("Step")
            ax0.set_ylabel("Color dist")
            ax0.legend(fontsize=6, loc='best')

            for color_id in range(num_colors):
                valid = elected_ranks[:, color_id] >= 0
                if np.any(valid):
                    ax1.plot(
                        steps[valid],
                        elected_ranks[valid, color_id],
                        marker="o",
                        label=f"Color {color_id}",
                        color=COLORS[color_id],
                        linewidth=0.2,
                    )
            ax1.set_title("Elected ordering")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Rank")
            ax1.invert_yaxis()

            hist = area.diag_history[-self.max_steps:]
            if hist:
                hist_len = len(hist)
                step_axis = np.arange(step - hist_len + 1, step + 1)
                group_personal = [h.get("group_mean_personal_reward", []) for h in hist]
                group_fee = [h.get("group_mean_fee", []) for h in hist]

                num_groups = len(group_personal[0]) if group_personal and group_personal[0] is not None else 0

                for g in range(num_groups):
                    p_series = _series_at(g, group_personal)
                    f_series = _series_at(g, group_fee)
                    color = get_group_color(g)

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
                step_axis = np.arange(step - hist_len + 1, step + 1)

                group_turnout = [h.get("group_turnout", []) for h in hist]
                overall_turnout = [h.get("turnout", float("nan")) for h in hist]
                group_assets = [h.get("group_mean_assets", []) for h in hist]
                group_delta_p = [h.get("group_mean_delta_rel_participants", []) for h in hist]
                group_delta_a = [h.get("group_mean_delta_rel_abstainers", []) for h in hist]

                num_groups = len(group_turnout[0]) if group_turnout and group_turnout[0] is not None else 0

                for g in range(num_groups):
                    t_series = _series_at(g, group_turnout)
                    a_series = _series_at(g, group_assets)
                    dp_series = _series_at(g, group_delta_p)
                    da_series = _series_at(g, group_delta_a)
                    color = get_group_color(g)

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
                step_axis = np.arange(step - hist_len + 1, step + 1)

                group_altruism = [h.get("group_mean_altruism", []) for h in hist]
                group_q_p = [h.get("group_mean_q_participation_participants", []) for h in hist]
                group_q_a = [h.get("group_mean_q_participation_abstainers", []) for h in hist]
                group_dissatisfaction = [h.get("group_mean_dissatisfaction", []) for h in hist]

                num_groups = len(group_altruism[0]) if group_altruism and group_altruism[0] is not None else 0

                for g in range(num_groups):
                    a_series = _series_at(g, group_altruism)
                    qp_series = _series_at(g, group_q_p)
                    qa_series = _series_at(g, group_q_a)
                    s_series = _series_at(g, group_dissatisfaction)
                    color = get_group_color(g)

                    ax6.plot(step_axis, a_series, color=color, label=f"g{g}")
                    ax7.plot(step_axis, qp_series, color=color, linestyle=":", label=f"g{g} p")
                    ax7.plot(step_axis, qa_series, color=color, linestyle=":", label=f"g{g} a")
                    ax8.plot(step_axis, s_series, color=color, label=f"g{g}")

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


class PersonalityGroupDistribution(TextElement):
    def __init__(self, *, collapsible: bool = True, default_open: bool = False):
        super().__init__()
        self.pers_dist_plot = None
        self.collapsible = collapsible
        self.default_open = default_open

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

        ax.set_xlabel('"Preference Group" ID')
        ax.set_ylabel(f'Percentage of the {num_agents} Agents')
        ax.set_title('Global distribution of preference groups among agents')
        plt.tight_layout()
        self.pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model) -> str:
        if model.scheduler.steps == 0:
            self.create_once(model)
        return wrap_html_panel(
            self.pers_dist_plot or "",
            title="Global preference-group distribution",
            collapsible=self.collapsible,
            default_open=self.default_open,
        )


class AreaPuzzleColorDistributionElement(TextElement):
    def __init__(self, *, collapsible: bool = True, default_open: bool = False):
        super().__init__()
        self.collapsible = collapsible
        self.default_open = default_open
        self._cached_step = -1
        self._cached_html = ""

    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == self._cached_step:
            return self._cached_html

        data = model.datacollector.get_agent_vars_dataframe()
        if data is None or data.empty or "puzzle_color_distribution" not in data.columns:
            return ""

        puzzle_rows = data[data["puzzle_color_distribution"].notna()]
        if puzzle_rows.empty:
            return ""

        area_frames = {
            area_id: frame
            for area_id, frame in puzzle_rows.groupby(level=1, sort=False)
        }
        area_ids = sorted(area_frames)
        if not area_ids:
            return ""

        sample = area_frames[area_ids[0]]["puzzle_color_distribution"].iloc[0]
        num_colors = len(sample)

        n_areas = len(area_ids)
        fig, axes = plt.subplots(nrows=n_areas, ncols=1, figsize=(10, max(3.0, 2.5 * n_areas)), sharex=True)
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, area_id in zip(axes_list, area_ids):
            area_frame = area_frames[area_id]
            steps = area_frame.index.get_level_values(0).to_numpy(dtype=float)
            puzzle_matrix = _vector_matrix(area_frame["puzzle_color_distribution"].tolist(), num_colors)
            for color_idx in range(num_colors):
                ax.plot(steps, puzzle_matrix[:, color_idx], color=COLORS[color_idx], linewidth=1.2, label=f"c{color_idx}")
            ax.set_ylabel(f"Area {area_id}")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2, linewidth=0.5)
            if num_colors <= 8:
                ax.legend(fontsize=7, loc="best")

        axes_list[-1].set_xlabel("Step")
        fig.suptitle("Puzzle Color Distribution by Area", fontsize=12)
        plt.tight_layout()
        self._cached_html = wrap_html_panel(
            save_plot_to_base64(fig),
            title="Puzzle color distribution by area",
            collapsible=self.collapsible,
            default_open=self.default_open,
        )
        self._cached_step = step
        return self._cached_html


class MainMetricsElement(TextElement):
    """Compact 2x2 overview of thesis metrics by preference group."""

    _METRICS = (
        ("turnout", "group_turnout", "Turnout and Turnout by Group", "overall turnout", "group", "Turnout (%)", None, "black", (0.0, 100.0), None),
        ("gini_index", "group_mean_assets_share", "Asset Gini and Relative Mean Assets by Group", "overall asset gini", "group", "Gini (0-100)", "Relative mean assets", "black", (0.0, 100.0), (0.0, 1.0)),
        ("gini_dissatisfaction", "group_mean_dissatisfaction", "Dissatisfaction Gini and Mean Dissatisfaction by Group", "overall dissatisfaction gini", "group", "Gini (0-100)", "Mean dissatisfaction", "black", (0.0, 100.0), (0.0, 1.0)),
        ("quality_distance", "group_outcome_distance", "Quality Distance and Group Distance to Outcome", "overall quality distance", "group", "Distance", None, "black", (0.0, 1.0), None),
    )

    def __init__(self, *, collapsible: bool = False, default_open: bool = True):
        super().__init__()
        self.collapsible = collapsible
        self.default_open = default_open
        self._cached_step = -1
        self._cached_html = ""

    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == self._cached_step:
            return self._cached_html

        model_vars = model.datacollector.model_vars
        if not model_vars:
            return ""

        required = {group_col for _, group_col, *_ in self._METRICS}
        if not required.issubset(model_vars):
            return ""

        n_groups = model.num_personality_groups
        if n_groups == 0:
            return ""

        num_steps = len(next(iter(model_vars.values())))
        x = _step_axis(model, model_vars, num_steps)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.5, 7.0), sharex=True)
        axes_flat = axes.flatten()
        has_step_zero = num_steps > 0 and x[0] == 0.0

        for ax, metric in zip(axes_flat, self._METRICS):
            overall_col, group_col, title, overall_label, group_label_prefix, left_ylabel, right_ylabel, overall_color, overall_ylim, group_ylim = metric
            group_mat = _vector_matrix(model_vars[group_col], n_groups)
            overall = None
            if overall_col in model_vars:
                overall = _float_array(model_vars[overall_col])

            if has_step_zero:
                if overall_col in {"turnout", "gini_dissatisfaction"}:
                    group_mat[0, :] = np.nan
                    if overall is not None:
                        overall[0] = np.nan
                elif overall_col == "quality_distance" and overall is not None:
                    overall[0] = np.nan

            any_group_line = False
            group_ax = ax.twinx() if group_ylim is not None else ax
            for g in range(n_groups):
                y = group_mat[:, g]
                if not np.isfinite(y).any():
                    continue
                any_group_line = True
                line_color = _with_alpha(get_group_color(g), 0.35 if overall_col == "quality_distance" else 0.45)
                group_ax.plot(x, y, color=line_color, linewidth=1.4, label=f"{group_label_prefix} {g}")

            if overall is not None:
                if np.isfinite(overall).any():
                    ax.plot(x, overall, color=overall_color, linewidth=1.25, linestyle="--", alpha=0.85, label=overall_label)
                    ax.set_title(title)
                else:
                    ax.set_title(title)
            else:
                ax.set_title(title)

            ax.set_ylim(*overall_ylim)
            ax.set_ylabel(left_ylabel)
            if group_ylim is not None:
                group_ax.set_ylim(*group_ylim)
                group_ax.set_yticks([])
                group_ax.set_ylabel(right_ylabel)
            ax.grid(alpha=0.2, linewidth=0.5)
            if any_group_line and n_groups <= 8:
                if group_ax is ax:
                    group_ax.legend(fontsize=7, loc="best")
                else:
                    overall_handles, overall_labels = ax.get_legend_handles_labels()
                    group_handles, group_labels = group_ax.get_legend_handles_labels()
                    group_ax.legend(overall_handles + group_handles, overall_labels + group_labels, fontsize=7, loc="best")

        axes[1][0].set_xlabel("Step")
        axes[1][1].set_xlabel("Step")
        plt.tight_layout()
        self._cached_html = wrap_html_panel(
            save_plot_to_base64(fig),
            title="Main metrics by preference group",
            collapsible=self.collapsible,
            default_open=self.default_open,
        )
        self._cached_step = step
        return self._cached_html

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
        return (f"Step: {step} | cells: {len(model.color_cells)} | "
                f"areas: {len(model.areas)} | First 5 voters of "
                f"{len(model.voting_agents)}: {first_agents}")


class ReplayGridStepStatusElement(TextElement):
    """Show replay step/grid alignment status (useful for sparse grid snapshots)."""

    def render(self, model) -> str:
        if not hasattr(model, "replay_recorded_step") or not hasattr(model, "replay_grid_source_step"):
            return ""

        rec = model.replay_recorded_step
        src = model.replay_grid_source_step

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
    def __init__(self, *, collapsible: bool = True, default_open: bool = False):
        super().__init__()
        self.areas_pers_dist_plot = None
        self.collapsible = collapsible
        self.default_open = default_open

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
            bar_colors = [get_group_color(i) for i in range(num_personality_groups)]
            bars = ax.bar(range(num_personality_groups), heights, color=bar_colors)
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
        return wrap_html_panel(
            self.areas_pers_dist_plot or "",
            title="Per-area preference-group distributions",
            collapsible=self.collapsible,
            default_open=self.default_open,
        )


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
        step = model.scheduler.steps
        agents = [a for a in model.voting_agents if a is not None]
        if not agents:
            return ""

        # Collect vectors (fail loudly if model/agent contract is broken)
        p = np.asarray([a.participation_probability() for a in agents], dtype=np.float64)
        q = np.asarray([a.q_participation for a in agents], dtype=np.float64)
        altruism = np.asarray([a.altruism_factor for a in agents], dtype=np.float64)
        assets = np.asarray([a.assets for a in agents], dtype=np.float64)

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

        p_mean = np.mean(p_f)
        p_median = np.median(p_f)
        a_mean = np.mean(a_f)
        a_median = np.median(a_f)

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

    Per preference group (cohort), we compute (eligible agents only):
      - counts + participation_rate
      - mean/median delta for participants vs abstainers
      - mean fee (participants), mean personal reward
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
        self.top_k = top_k

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
            # Has to be robust to account for replay agent-stubs
            eligible = a.eligible_for_election
            if not eligible:
                continue
            gid_i = a.personality_group_idx

            participated = a.participating
            delta = a.election_delta_abs
            fee = getattr(a, "_fee")
            personal = a.reward_personal
            altruism = a.altruism_factor
            try:
                p_part = a.participation_probability()
            except ValueError:
                p_part = float("nan")

            rows.append(
                (gid_i, participated, delta, fee, personal, altruism, p_part)
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

        labels = [str(g) for g in show_groups]
        if show_other:
            labels.append("other")

        group_ids = [g for g in show_groups]
        if show_other:
            group_ids.append(-1)

        def _mask_for_group(gval):
            return gid == gval

        # Aggregate per shown group
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

        def _add_bucket(mask: np.ndarray):
            elig_n = int(np.sum(mask))
            part_mask = mask & participated
            abst_mask = mask & (~participated)
            part_n = int(np.sum(part_mask))
            abst_n = int(np.sum(abst_mask))
            rate = part_n / elig_n if elig_n > 0 else float("nan")

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
            if k == "group_ids":
                out[k] = np.asarray(out[k], dtype=np.int64)
                continue
            out[k] = np.asarray(out[k], dtype=np.float64)

        return out

    def render(self, model) -> str:
        step = model.scheduler.steps
        if step == 0:
            # Avoid noisy empty plots before the first election has happened.
            return ""

        stats = self._compute(model)
        if stats is None:
            return ""

        labels = stats["labels"]
        group_ids = stats.get("group_ids", list(range(len(labels))))
        n = len(labels)
        x = np.arange(n)
        group_colors = [get_group_color(g) for g in group_ids]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))

        # Panel A: participation rate
        ax = axes[0][0]
        rate = stats["rate"]
        ax.bar(x, np.nan_to_num(rate, nan=0.0), color=group_colors, alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Participation rate by preference group")
        ax.set_ylabel("participant_count / eligible")

        # Panel B: mean delta participants vs abstainers
        ax = axes[0][1]
        w = 0.4
        dp = stats["delta_p_mean"]
        da = stats["delta_a_mean"]
        ax.bar(x - w / 2, np.nan_to_num(dp, nan=0.0), width=w, label="participants", color=group_colors, alpha=0.8)
        ax.bar(x + w / 2, np.nan_to_num(da, nan=0.0), width=w, label="abstainers", color=group_colors, alpha=0.4)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Mean delta_assets by action (eligible only)")
        ax.set_ylabel("mean delta_assets")
        ax.legend()

        # Panel C: decomposition
        ax = axes[1][0]
        fee_m = stats["fee_mean"]
        pers_m = stats["personal_mean"]
        ax.bar(x - w, np.nan_to_num(fee_m, nan=0.0), width=w, label="fee (participants)", color=group_colors, alpha=0.8)
        ax.bar(x, np.nan_to_num(pers_m, nan=0.0), width=w, label="personal reward", color=group_colors, alpha=0.6)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Reward/Fee means")
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

        elig = stats["eligible"]
        parts = stats["participants"]
        abst = stats["abstainers"]
        fig.suptitle(
            f"Cohort election learning diagnostics (step={step}) | eligible/part/abst per group: "
            + ", ".join([f"{labels[i]}:{elig[i]}/{parts[i]}/{abst[i]}" for i in range(n)])
        )
        plt.tight_layout()
        return save_plot_to_base64(fig)
    
