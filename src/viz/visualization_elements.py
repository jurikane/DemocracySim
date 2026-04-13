from __future__ import annotations

import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mesa.visualization import TextElement

from src.viz.color_palette import get_group_color
from src.viz.factory import COLORS
from src.viz.helpers import (
    auto_ylim,
    float_array,
    save_plot_to_base64,
    step_axis,
    vector_matrix,
    wrap_html_panel,
)

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

        fig, ax = plt.subplots(figsize=(4, 3.0))
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
        plt.tight_layout()
        self.pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model) -> str:
        if model.scheduler.steps == 0:
            self.create_once(model)
        return wrap_html_panel(
            self.pers_dist_plot or "",
            title="Global distribution of preference groups among agents",
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
        fig, axes = plt.subplots(nrows=n_areas, ncols=1, figsize=(9, max(3.0, 2.5 * n_areas)), sharex=True)
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, area_id in zip(axes_list, area_ids):
            area_frame = area_frames[area_id]
            steps = area_frame.index.get_level_values(0).to_numpy(dtype=float)
            puzzle_matrix = vector_matrix(area_frame["puzzle_color_distribution"].tolist(), num_colors)
            for color_idx in range(num_colors):
                ax.plot(steps, puzzle_matrix[:, color_idx], color=COLORS[color_idx], linewidth=1.2)
            ax.set_ylabel(f"Area {area_id}")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2, linewidth=0.5)

        axes_list[-1].set_xlabel("Step")
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
    """Compact thesis-metric panels by preference group."""

    def __init__(self, *, collapsible: bool = True, default_open: bool = True):
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

        required = {
            "group_turnout",
            "group_mean_assets_share",
            "group_mean_dissatisfaction",
            "group_outcome_distance",
        }
        if not required.issubset(model_vars):
            return ""

        n_groups = model.num_personality_groups
        if n_groups == 0:
            return ""

        num_steps = len(next(iter(model_vars.values())))
        x = step_axis(model, model_vars, num_steps)
        has_step_zero = x.size > 0 and x[0] == 0.0

        def series(overall_col: str, group_col: str, *, max_steps: int | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
            overall = float_array(model_vars[overall_col]) if overall_col in model_vars else None
            group_mat = vector_matrix(model_vars[group_col], n_groups)

            if has_step_zero:
                if overall_col in {"turnout", "gini_dissatisfaction"}:
                    group_mat[0, :] = np.nan
                    if overall is not None:
                        overall[0] = np.nan
                elif overall_col == "quality_distance" and overall is not None:
                    overall[0] = np.nan

            if max_steps is None or x.size <= max_steps:
                return x, overall, group_mat

            start = x.size - max_steps
            return x[start:], None if overall is None else overall[start:], group_mat[start:, :]

        def plot_metric(
            ax,
            *,
            overall_col: str,
            group_col: str,
            title: str,
            left_ylabel: str,
            right_ylabel: str | None = None,
            overall_ylim: tuple[float, float],
            group_ylim: tuple[float, float] | None = None,
            max_steps: int | None = None,
            legend_loc: str = "best",
        ) -> None:
            plot_x, overall, group_mat = series(overall_col, group_col, max_steps=max_steps)
            group_ax = ax if group_ylim is None else ax.twinx()
            if group_ax is not ax:
                ax.set_zorder(group_ax.get_zorder() + 1)
                ax.patch.set_visible(False)
            has_group_lines = False

            for group_idx in range(n_groups):
                values = group_mat[:, group_idx]
                if not np.isfinite(values).any():
                    continue
                has_group_lines = True
                group_ax.plot(
                    plot_x,
                    values,
                    color=get_group_color(group_idx),
                    linewidth=1.2,
                    label=f"group {group_idx}",
                    zorder=2,
                )

            if overall is not None and np.isfinite(overall).any():
                ax.plot(
                    plot_x,
                    overall,
                    color="black",
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.85,
                    label=f"overall {overall_col.replace('_', ' ')}",
                    zorder=10,
                )

            ax.set_title(title)
            if overall_col == "gini_index" and overall is not None:
                ax.set_ylim(*auto_ylim(overall, lower_bound=0.0, upper_bound=100.0))
            else:
                ax.set_ylim(*overall_ylim)
            ax.set_ylabel(left_ylabel)
            ax.set_xlabel("Step")
            ax.grid(alpha=0.2, linewidth=0.5)

            if group_ylim is not None:
                group_ax.set_ylim(*group_ylim)
                group_ax.set_yticks([])
                group_ax.set_ylabel(right_ylabel)

            if has_group_lines and n_groups <= 8:
                if group_ax is ax:
                    group_ax.legend(fontsize=7, loc=legend_loc)
                else:
                    overall_handles, overall_labels = ax.get_legend_handles_labels()
                    group_handles, group_labels = group_ax.get_legend_handles_labels()
                    group_ax.legend(overall_handles + group_handles, overall_labels + group_labels, fontsize=7, loc=legend_loc)

        fig_turnout, ax_turnout = plt.subplots(figsize=(9.0, 3.5))
        plot_metric(
            ax_turnout,
            overall_col="turnout",
            group_col="group_turnout",
            title="Turnout and Turnout by Group",
            left_ylabel="Turnout (%)",
            overall_ylim=(0.0, 100.0),
            max_steps=80,
            legend_loc="lower left",
        )

        fig_inequality, ax_inequality = plt.subplots(ncols=2, figsize=(9.0, 3.5), sharex=True)
        plot_metric(
            ax_inequality[0],
            overall_col="gini_index",
            group_col="group_mean_assets_share",
            title="Asset Gini and Relative Mean Assets by Group",
            left_ylabel="Gini (0-100)",
            right_ylabel="Relative mean assets",
            overall_ylim=(0.0, 100.0),
            group_ylim=(0.0, 1.0),
            legend_loc="upper left",
        )
        plot_metric(
            ax_inequality[1],
            overall_col="gini_dissatisfaction",
            group_col="group_mean_dissatisfaction",
            title="Dissatisfaction: Overall Gini and Mean by Group",
            left_ylabel="Gini (0-100)",
            right_ylabel="Mean dissatisfaction",
            overall_ylim=(0.0, 100.0),
            group_ylim=(0.0, 1.0),
            legend_loc="upper left",
        )

        fig_quality, ax_quality = plt.subplots(figsize=(9.0, 3.5))
        plot_metric(
            ax_quality,
            overall_col="quality_distance",
            group_col="group_outcome_distance",
            title="Outcome Quality and Group Distance to Outcome",
            left_ylabel="Distance",
            overall_ylim=(0.0, 1.0),
            max_steps=50,
            legend_loc="upper left",
        )

        fig_turnout.tight_layout()
        fig_inequality.tight_layout()
        fig_quality.tight_layout()
        self._cached_html = "".join(
            (
                wrap_html_panel(
                    save_plot_to_base64(fig_turnout),
                    title="Turnout",
                    collapsible=self.collapsible,
                    default_open=self.default_open,
                ),
                wrap_html_panel(
                    save_plot_to_base64(fig_inequality),
                    title="Inequality",
                    collapsible=self.collapsible,
                    default_open=self.default_open,
                ),
                wrap_html_panel(
                    save_plot_to_base64(fig_quality),
                    title="Outcome quality",
                    collapsible=self.collapsible,
                    default_open=self.default_open,
                ),
            )
        )
        self._cached_step = step
        return self._cached_html


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
    def __init__(self, *, collapsible: bool = True, default_open: bool = True):
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
