from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba

from src.analysis.summary_io import _load_grid_with_carry_forward
from src.analysis.quality_distance import quality_distance_source
from src.analysis.summary_render_area import (
    _draw_area_personality_group_distribution,
    _draw_personality_group_order_block,
    _draw_reference_optima_panel,
    _sim_color,
)
from src.analysis.summary_render_common import _adjacent_abs_change_series, _rolling_mean_nan, _set_percent_ylim_visible, _set_unit_ylim_visible
from src.analysis.thesis_endpoints import step_volatility_l1_normalized
from src.viz.color_palette import get_group_color

_SMOOTH_WINDOW_STEPS = 9

def _render_static_overview_pdf(*, out_pdf: Path, static: dict[str, Any], meta: dict[str, Any]) -> None:
    with PdfPages(out_pdf) as pdf:
        _append_static_overview_pages(pdf=pdf, static=static, meta=meta)

def _append_static_overview_pages(*, pdf: PdfPages, static: dict[str, Any], meta: dict[str, Any]) -> None:
    num_colors = int(static.get("num_colors", 0))
    num_areas = int(static.get("num_areas", 0))
    num_agents = int(static.get("num_agents", 0))
    width = int(static.get("width", 0))
    height = int(static.get("height", 0))
    run_seed = int(meta["run"]["run_seed"])
    rule_name = meta["run"].get("rule_name")
    distance_name = meta["run"].get("distance_name")

    info = static.get("personality_group_info", {}) or {}
    personality_groups = np.asarray(info.get("personality_groups", []), dtype=int)
    global_dist = np.asarray(info.get("global_distribution", []), dtype=float)
    areas_info = info.get("areas", {}) or {}

    if personality_groups.ndim != 2:
        personality_groups = np.zeros((0, num_colors), dtype=int)
    n_groups = int(personality_groups.shape[0])

    # Single-page layout:
    # left-top: personality order with color blocks
    # left-bottom: global group distribution
    # right: per-area group composition (full height)
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.6])
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.2, 0.8])

    ax_map = fig.add_subplot(left[0, 0])
    ax_global = fig.add_subplot(left[1, 0])
    ax_area = fig.add_subplot(outer[0, 1])

    fig.suptitle(
        f"Static Overview | run_seed={run_seed} | rule={rule_name} | distance={distance_name} | "
        f"grid={width}x{height} | agents={num_agents} | areas={num_areas} | colors={num_colors}",
        fontsize=11,
    )

    _draw_personality_group_order_block(
        ax=ax_map,
        personality_groups=personality_groups,
        num_colors=num_colors,
    )

    # Global group distribution
    if n_groups > 0 and global_dist.size == n_groups:
        x = np.arange(n_groups)
        # Background: per-group preference-order stripes (same idea as ordering bands in area quality-distance plots).
        if personality_groups.ndim == 2 and personality_groups.shape[0] >= n_groups:
            n_slots = int(min(num_colors, personality_groups.shape[1]))
            for gi in range(n_groups):
                order = personality_groups[gi].astype(int).tolist()
                for rank, color_id in enumerate(order[:n_slots]):
                    y0 = 1.0 - float(rank + 1) / float(n_slots)
                    ax_global.add_patch(
                        plt.Rectangle(
                            (float(gi) - 0.4, y0),
                            0.8,
                            1.0 / float(n_slots),
                            facecolor=_sim_color(color_id),
                            edgecolor="none",
                            alpha=0.26,
                            zorder=0,
                        )
                    )
        # Foreground: transparent bars (black frames only).
        ax_global.bar(
            x,
            global_dist,
            width=0.8,
            facecolor="none",
            edgecolor=[get_group_color(i) for i in range(n_groups)],
            linewidth=1.2,
            zorder=2,
        )
        # Label shares at bar tops; if there is no space above, place just below.
        for gi, val in enumerate(global_dist.tolist()):
            y = float(val)
            label = f"{100.0 * y:.1f}%"
            if y <= 0.93:
                ax_global.text(
                    float(gi),
                    y + 0.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=get_group_color(int(gi)),
                    fontweight="bold",
                )
            else:
                ax_global.text(
                    float(gi),
                    y - 0.03,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=get_group_color(int(gi)),
                    fontweight="bold",
                )
        ax_global.set_xticks(x)
        ax_global.set_xticklabels([f"g{i}" for i in range(n_groups)])
        # Encode group-color mapping directly in the x-axis labels.
        for gi, tick in enumerate(ax_global.get_xticklabels()):
            c = get_group_color(int(gi))
            r, g, b, _ = to_rgba(c)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt = "black" if luminance > 0.55 else "white"
            tick.set_color(txt)
            tick.set_bbox(
                dict(
                    facecolor=c,
                    edgecolor="none",  # frameless colored square-ish tag
                    boxstyle="round,pad=0.20,rounding_size=0.08",
                    alpha=0.95,
                )
            )
        _set_unit_ylim_visible(ax_global)
        if n_groups > 0:
            major = int(np.argmax(global_dist))
            ax_global.text(
                0.99,
                0.98,
                f"majority: g{major} ({100.0 * float(global_dist[major]):.1f}%)",
                transform=ax_global.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
    else:
        ax_global.text(0.5, 0.5, "No global group metadata", ha="center", va="center")
    ax_global.set_title("Personality Groups with their Global Shares")
    ax_global.set_yticks([])
    ax_global.set_ylabel("")
    ax_global.grid(True, axis="y", alpha=0.25)

    # Per-area group composition
    area_rows: list[tuple[str, int, np.ndarray]] = []
    for area_key in sorted(areas_info.keys(), key=lambda x: int(x)):
        payload = areas_info.get(area_key) or {}
        area_n = int(payload.get("num_agents", 0))
        dist = np.asarray(payload.get("personality_group_distribution", []), dtype=float)
        area_rows.append((area_key, area_n, dist))

    if area_rows and n_groups > 0:
        n_area = len(area_rows)
        if n_area < 16:
            # Keep visual density comparable to larger-area runs:
            # center rows inside a virtual 16-row frame.
            y_offset = 0.5 * (16 - n_area)
            y = np.arange(n_area, dtype=float) + y_offset
            bar_h = 0.55
            ax_area.set_ylim(-0.5, 15.5)
        else:
            y = np.arange(n_area, dtype=float)
            bar_h = 0.8
        left_vals = np.zeros(len(area_rows), dtype=float)
        for gi in range(n_groups):
            vals = np.array(
                [float(r[2][gi]) if r[2].size > gi else 0.0 for r in area_rows],
                dtype=float,
            )
            ax_area.barh(
                y,
                vals,
                left=left_vals,
                height=bar_h,
                label=f"g{gi}",
                color=get_group_color(gi),
            )
            left_vals += vals
        labels = [f"a{a} (n={n})" for a, n, _ in area_rows]
        ax_area.set_yticks(y)
        ax_area.set_yticklabels(labels)
        ax_area.set_xlim(0.0, 1.0)
        ax_area.legend(loc="lower right", fontsize=8, ncol=2)
    else:
        ax_area.text(0.5, 0.5, "No area group metadata", ha="center", va="center")
    ax_area.set_title("Per-Area Group Composition")
    ax_area.set_xlabel("share")
    ax_area.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)

def _render_combined_global_summary_pdf(
    *,
    out_pdf: Path,
    run_dir: Path,
    global_series: pd.DataFrame,
    steps: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_global: dict[str, np.ndarray | None],
    render_profile: _SummaryRenderProfile,
) -> None:
    with PdfPages(out_pdf) as pdf:
        if render_profile.global_colors_and_grids:
            _render_global_colors_and_grids_page(
                pdf=pdf,
                run_dir=run_dir,
                global_series=global_series,
                steps=steps,
                static=static,
                refs_global=refs_global,
            )
        if render_profile.global_static_overview:
            _append_static_overview_pages(pdf=pdf, static=static, meta=meta)
        if render_profile.global_per_area_group_distribution:
            _append_per_area_group_distribution_pages(pdf=pdf, static=static, meta=meta)
        if render_profile.global_core_metrics:
            _render_global_core_metrics_page(pdf=pdf, global_series=global_series, meta=meta, static=static)
        if render_profile.global_step_volatility_page:
            _render_global_step_volatility_page(pdf=pdf, global_series=global_series, meta=meta, static=static)
        if render_profile.global_distance_metrics:
            _render_global_distance_page(pdf=pdf, global_series=global_series, meta=meta)

def _append_per_area_group_distribution_pages(*, pdf: PdfPages, static: dict[str, Any], meta: dict[str, Any]) -> None:
    """Append area-wise group-distribution pages using the same visual style as global."""
    num_colors = int(static.get("num_colors", 0))
    info = static.get("personality_group_info", {}) or {}
    personality_groups = np.asarray(info.get("personality_groups", []), dtype=int)
    areas_info = info.get("areas", {}) or {}
    if personality_groups.ndim != 2:
        return
    n_groups = int(personality_groups.shape[0])
    if n_groups <= 0 or not isinstance(areas_info, dict) or len(areas_info) == 0:
        return

    rows: list[tuple[int, int, np.ndarray]] = []
    for area_key in sorted(areas_info.keys(), key=lambda x: int(x)):
        payload = areas_info.get(area_key) or {}
        dist = np.asarray(payload.get("personality_group_distribution", []), dtype=float)
        if dist.size != n_groups:
            continue
        rows.append((int(area_key), int(payload.get("num_agents", 0)), dist))
    if not rows:
        return

    run_seed = int(meta["run"]["run_seed"])
    rule_name = meta["run"].get("rule_name")
    per_page = 9
    n_pages = int(np.ceil(len(rows) / per_page))

    for p in range(n_pages):
        chunk = rows[p * per_page:(p + 1) * per_page]
        fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27))
        ax_list = axes.ravel()
        for idx, ax in enumerate(ax_list):
            if idx >= len(chunk):
                ax.axis("off")
                continue
            area_id, n_agents, dist = chunk[idx]
            x = np.arange(n_groups)
            # Background ordering stripes per group.
            n_slots = int(min(num_colors, personality_groups.shape[1]))
            for gi in range(n_groups):
                order = personality_groups[gi].astype(int).tolist()
                for rank, color_id in enumerate(order[:n_slots]):
                    y0 = 1.0 - float(rank + 1) / float(n_slots)
                    ax.add_patch(
                        plt.Rectangle(
                            (float(gi) - 0.4, y0),
                            0.8,
                            1.0 / float(n_slots),
                            facecolor=_sim_color(color_id),
                            edgecolor="none",
                            alpha=0.26,
                            zorder=0,
                        )
                    )
            # Transparent bars with black frame.
            ax.bar(
                x,
                dist,
                width=0.8,
                facecolor="none",
                edgecolor=[get_group_color(i) for i in range(n_groups)],
                linewidth=1.1,
                zorder=2,
            )
            # Percent labels.
            for gi, v in enumerate(dist.tolist()):
                y = float(v)
                txt = f"{100.0 * y:.0f}%"
                if y <= 0.92:
                    ax.text(
                        float(gi),
                        y + 0.02,
                        txt,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=get_group_color(int(gi)),
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        float(gi),
                        y - 0.03,
                        txt,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color=get_group_color(int(gi)),
                        fontweight="bold",
                    )
            ax.set_title(f"Area {area_id} (n={n_agents})", fontsize=9)
            _set_unit_ylim_visible(ax)
            ax.set_yticks([])
            ax.set_xticks(x)
            ax.set_xticklabels([f"g{i}" for i in range(n_groups)], fontsize=7)
            for gi, tick in enumerate(ax.get_xticklabels()):
                c = get_group_color(int(gi))
                r, g, b, _ = to_rgba(c)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                txt = "black" if luminance > 0.55 else "white"
                tick.set_color(txt)
                tick.set_bbox(
                    dict(
                        facecolor=c,
                        edgecolor="none",
                        boxstyle="round,pad=0.14,rounding_size=0.06",
                        alpha=0.95,
                    )
                )
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(
            f"Per-Area Personality Group Distributions | run_seed={run_seed} | rule={rule_name} | page {p + 1}/{n_pages}",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

def _render_global_core_metrics_page(
    *,
    pdf: PdfPages,
    global_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    ax = axes.ravel()
    ax[0].plot(x, global_series["turnout"].to_numpy(dtype=float), color="tab:blue")
    ax[0].set_title("Turnout [%]")
    _set_percent_ylim_visible(ax[0])
    ax[1].plot(x, global_series["gini_assets"].to_numpy(dtype=float), color="tab:red")
    ax[1].set_title("Gini Assets [0..100]")
    _set_percent_ylim_visible(ax[1])
    ax[2].plot(x, global_series["gini_dissatisfaction"].to_numpy(dtype=float), color="tab:purple")
    ax[2].set_title("Gini Dissatisfaction [0..100]")
    _set_percent_ylim_visible(ax[2])
    ax[3].plot(x, global_series["mean_dissatisfaction"].to_numpy(dtype=float), color="tab:orange")
    ax[3].set_title("Mean Dissatisfaction")
    _set_unit_ylim_visible(ax[3])

    vol_lines = [
        (
            "turnout_volatility",
            step_volatility_l1_normalized(global_series["turnout"].to_numpy(dtype=float), value_range=100.0),
        ),
        (
            "gini_assets_volatility",
            step_volatility_l1_normalized(global_series["gini_assets"].to_numpy(dtype=float), value_range=100.0),
        ),
        (
            "gini_dissatisfaction_volatility",
            step_volatility_l1_normalized(global_series["gini_dissatisfaction"].to_numpy(dtype=float), value_range=100.0),
        ),
        (
            "quality_distance_volatility",
            step_volatility_l1_normalized(global_series["quality_distance"].to_numpy(dtype=float), value_range=1.0),
        ),
    ]
    vol_text = "Adjacent-step volatility (mean |Δ|)\n" + "\n".join(
        f"{k}: {v:.3f}" if np.isfinite(v) else f"{k}: nan" for k, v in vol_lines
    )
    ax[3].text(
        0.02,
        0.98,
        vol_text,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.5", alpha=0.9),
        transform=ax[3].transAxes,
    )
    for a in ax:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    num_areas = int(static.get("num_areas", 0))
    scope = "Global (= area_0 aggregate; n_areas=1)" if num_areas == 1 else f"Global aggregate (n_areas={num_areas})"
    title = f"Global Core Metrics | run_seed={meta['run']['run_seed']} | rule={meta['run'].get('rule_name')}\n{scope}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)

def _render_global_step_volatility_page(
    *,
    pdf: PdfPages,
    global_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    rows = [
        ("turnout", "Turnout [%]", "tab:blue", "percent"),
        ("gini_assets", "Gini Assets [0..100]", "tab:red", "percent"),
        ("gini_dissatisfaction", "Gini Dissatisfaction [0..100]", "tab:purple", "percent"),
        ("quality_distance", "quality_distance [0..1]", "tab:green", "unit"),
    ]
    for r, (col, title, color, scale_kind) in enumerate(rows):
        y = global_series[col].to_numpy(dtype=float)
        delta = _adjacent_abs_change_series(y)
        left = axes[r, 0]
        right = axes[r, 1]

        left.plot(x, y, color=color, linewidth=1.8)
        left.set_title(f"{title} level")
        if scale_kind == "percent":
            _set_percent_ylim_visible(left)
        else:
            _set_unit_ylim_visible(left)
        left.set_ylabel("value")
        left.grid(True, alpha=0.25)

        right.plot(x, delta, color=color, linewidth=0.95, alpha=0.25, label="|Δ| per step")
        right.plot(
            x,
            _rolling_mean_nan(delta),
            color=color,
            linewidth=1.6,
            alpha=0.95,
            label=f"rolling mean ({_SMOOTH_WINDOW_STEPS})",
        )
        mean_delta = float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan
        if np.isfinite(mean_delta):
            right.axhline(
                mean_delta,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label=f"mean |Δ| = {mean_delta:.3f}",
            )
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size > 0:
            ymax = float(np.nanmax(finite_delta))
            right.set_ylim(0.0, max(1e-6, 1.12 * ymax))
        else:
            right.set_ylim(0.0, 1.0)
        right.set_title(f"{title} adjacent-step volatility (mean |Δ|)")
        right.set_ylabel("|Δ|")
        right.grid(True, alpha=0.25)
        right.legend(loc="upper right", fontsize=7)

    for a in axes[-1, :]:
        a.set_xlabel("step")

    num_areas = int(static.get("num_areas", 0))
    scope = "Global (= area_0 aggregate; n_areas=1)" if num_areas == 1 else f"Global aggregate (n_areas={num_areas})"
    fig.suptitle(
        f"Global Step Volatility | run_seed={meta['run']['run_seed']} | rule={meta['run'].get('rule_name')}\n{scope}",
        fontsize=12,
    )
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)

def _render_global_distance_page(*, pdf: PdfPages, global_series: pd.DataFrame, meta: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    ax = axes.ravel()
    quality_mode = str(meta["run"].get("quality_target_mode", "reality"))
    source_col = quality_distance_source(quality_mode)
    ax[0].plot(x, global_series["quality_distance"].to_numpy(dtype=float), color="black", linewidth=1.8, label="quality_distance")
    if source_col == "puzzle_distance":
        if "dist_to_reality" in global_series.columns:
            ax[0].plot(
                x,
                global_series["dist_to_reality"].to_numpy(dtype=float),
                color="tab:green",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
                label="dist_to_reality",
            )
        if "puzzle_distance" in global_series.columns:
            ax[0].plot(
                x,
                global_series["puzzle_distance"].to_numpy(dtype=float),
                color="tab:blue",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="puzzle_distance (source)",
            )
    else:
        if "puzzle_distance" in global_series.columns:
            ax[0].plot(
                x,
                global_series["puzzle_distance"].to_numpy(dtype=float),
                color="tab:blue",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
                label="puzzle_distance",
            )
    ax[0].set_title(f"quality_distance (source={source_col})")
    _set_unit_ylim_visible(ax[0])
    ax[0].legend(loc="best", fontsize=8)
    ax[1].plot(x, global_series["dist_to_ref_utilitarian"].to_numpy(dtype=float), color="tab:blue", label="utilitarian")
    if "dist_to_ref_nash" in global_series.columns:
        ax[1].plot(x, global_series["dist_to_ref_nash"].to_numpy(dtype=float), color="tab:purple", label="nash")
    ax[1].plot(x, global_series["dist_to_ref_egalitarian"].to_numpy(dtype=float), color="tab:orange", label="egalitarian")
    ax[1].plot(x, global_series["dist_to_ref_rawlsian"].to_numpy(dtype=float), color="tab:red", label="rawlsian")
    ax[1].set_title("dist_to_ref_*")
    _set_unit_ylim_visible(ax[1])
    ax[1].legend(loc="best", fontsize=8)
    ax[2].plot(x, global_series["diversity_first_choice_entropy"].to_numpy(dtype=float), color="tab:brown")
    ax[2].set_title("diversity_first_choice_entropy")
    _set_unit_ylim_visible(ax[2])
    ax[3].axis("off")
    ax[3].text(
        0.02,
        0.98,
        "Distance metrics are lower-better.\nDiversity entropy is normalized to [0,1].\nNaN means no participants for that step.",
        va="top",
        ha="left",
        fontsize=10,
    )
    for a in ax[:3]:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig.suptitle("Global Quality Distance + Diversity Diagnostics", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)

def _render_global_colors_and_grids_page(
    *,
    pdf: PdfPages,
    run_dir: Path,
    global_series: pd.DataFrame,
    steps: pd.DataFrame,
    static: dict[str, Any],
    refs_global: dict[str, np.ndarray | None],
) -> None:
    num_colors = int(static.get("num_colors", 0))
    color_cols = [f"color_{i}" for i in range(num_colors) if f"color_{i}" in global_series.columns]

    refs = {
        "utilitarian": refs_global.get("dist_to_ref_utilitarian"),
        "nash": refs_global.get("dist_to_ref_nash"),
        "egalitarian": refs_global.get("dist_to_ref_egalitarian"),
        "rawlsian": refs_global.get("dist_to_ref_rawlsian"),
    }

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], width_ratios=[1.1, 1.4, 1.4])
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1:])
    x = global_series["step"].to_numpy(dtype=float)
    for i, c in enumerate(color_cols):
        ax_curve.plot(
            x,
            global_series[c].to_numpy(dtype=float),
            label=f"color_{i}",
            color=_sim_color(i),
        )
    ax_curve.set_title("Global Color Distribution Curves")
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("share")
    _set_unit_ylim_visible(ax_curve)
    ax_curve.grid(True, alpha=0.25)
    if color_cols:
        ax_curve.legend(loc="upper right", ncol=min(5, len(color_cols)), fontsize=8)
    _draw_reference_optima_panel(ax=ax_ref, refs=refs, num_colors=num_colors)

    # Grid snapshots: step 1 and last step (carry-forward if sparse interval).
    step_last = int(steps["step"].max()) if len(steps) > 0 else 1
    grid1 = _load_grid_with_carry_forward(run_dir=run_dir, step=1, max_step=step_last)
    grid_last = _load_grid_with_carry_forward(run_dir=run_dir, step=step_last, max_step=step_last)
    ax_note = fig.add_subplot(gs[1, 0])
    ax_g1 = fig.add_subplot(gs[1, 1])
    ax_gn = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.98,
        "Reference panel (top-left):\n"
        "four fixed benchmark distributions\n"
        "shown as colored dots per reference\n"
        "(y-axis = share 0..1),\n"
        "used by dist_to_ref_*.\n\n"
        "Color IDs and hues match simulation colors,\n"
        "so composition can be compared directly\n"
        "to time-varying global color curves.",
        va="top",
        ha="left",
        fontsize=9,
    )
    _draw_grid_or_note(ax=ax_g1, grid=grid1, title="Grid Snapshot @ step 1")
    _draw_grid_or_note(ax=ax_gn, grid=grid_last, title=f"Grid Snapshot @ step {step_last}")

    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)

def _draw_grid_or_note(*, ax, grid: np.ndarray | None, title: str) -> None:
    ax.set_title(title)
    if grid is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Grid snapshot not available", ha="center", va="center")
        return
    palette = [_sim_color(i) for i in range(int(np.nanmax(grid)) + 1)]
    cmap = ListedColormap(palette)
    bounds = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(grid, interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
