from __future__ import annotations

from pathlib import Path
from typing import Any
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.lines import Line2D

from src.analysis.summary_io import _load_required_finite_float_for_run
from src.analysis.quality_distance import quality_distance_source
from src.analysis.summary_render_common import _rolling_mean_nan, _set_percent_ylim_visible, _set_unit_ylim_visible
from src.analysis.summary_series import (
    _MODE_ALIGNMENT_LOW_SUPPORT_VOTES,
    _SMALL_GROUP_MIN_RESIDENTS,
    _summary_ordering_distance_func,
    _summary_ordering_from_distribution_tie_aware,
)
from src.analysis.doe_scoring import DEFAULT_SCORING_THRESHOLDS
from src.utils.ballots import score_options_c2
from src.utils.social_welfare_functions import approval_voting, borda_rule, majority_rule, schulze_rule, utilitarian_rule
from src.viz.color_palette import COLORS as SIM_COLORS
from src.viz.color_palette import get_group_color

_Y_PAD_PERCENT = 1.5


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _run_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(meta.get("run")) if isinstance(meta, dict) else {}


def _int_with_default(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str) and value.strip() == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _render_area_detail_pdfs(
    *,
    run_dir: Path,
    out_dir: Path,
    area_series: pd.DataFrame,
    area_group_series: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_by_area: dict[int, dict[str, np.ndarray | None]] | None = None,
    render_profile: _SummaryRenderProfile,
) -> None:
    area_ids = sorted(set(int(v) for v in area_series["area_id"].dropna().tolist()))
    for area_id in area_ids:
        block = area_series[area_series["area_id"].astype(int) == int(area_id)].sort_values("step").reset_index(drop=True)
        group_block = area_group_series[area_group_series["area_id"].astype(int) == int(area_id)].sort_values(["step", "group_idx"]).reset_index(drop=True)
        out_pdf = out_dir / f"area_{int(area_id)}.pdf"
        _render_area_detail_pdf(
            run_dir=run_dir,
            out_pdf=out_pdf,
            area_id=int(area_id),
            area_series=block,
            area_group_series=group_block,
            static=static,
            meta=meta,
            refs_area=(refs_by_area or {}).get(int(area_id), {}),
            render_profile=render_profile,
        )

def _render_area_detail_pdf(
    *,
    run_dir: Path,
    out_pdf: Path,
    area_id: int,
    area_series: pd.DataFrame,
    area_group_series: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_area: dict[str, np.ndarray | None],
    render_profile: _SummaryRenderProfile,
) -> None:
    with PdfPages(out_pdf) as pdf:
        puzzle_threshold = _load_required_finite_float_for_run(
            run_dir=run_dir,
            field="break_even_distance_common",
        )

        x = area_series["step"].to_numpy(dtype=float)
        participants = area_series["participants"].to_numpy(dtype=float)
        eligible = area_series["eligible_voters"].to_numpy(dtype=float)
        turnout = area_series["turnout"].to_numpy(dtype=float)

        # Compact static info block for area context.
        pgi = _as_dict(static.get("personality_group_info"))
        areas_info = _as_dict(pgi.get("areas"))
        area_info = _as_dict(areas_info.get(str(int(area_id))))
        area_n = _int_with_default(area_info.get("num_agents"), -1)
        pg_dist = np.asarray(area_info.get("personality_group_distribution", []), dtype=float)
        personality_groups = np.asarray(pgi.get("personality_groups", []), dtype=int)
        run_meta = _run_meta(meta)
        majority_txt = "n/a"
        if pg_dist.size > 0 and np.isfinite(pg_dist).any():
            gidx = int(np.nanargmax(pg_dist))
            majority_txt = f"g{gidx} ({100.0 * float(pg_dist[gidx]):.1f}%)"
        suptitle = (
            f"Area {area_id} Detail | run_seed={run_meta.get('run_seed')} | "
            f"rule={run_meta.get('rule_name')} | area_agents={area_n} | majority_group={majority_txt}"
        )
        current_rule_idx = _int_with_default(run_meta.get("rule_idx"), -1)
        power_dirs = _compute_area_power_direction_orderings(
            area_group_series=area_group_series,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
            meta=meta,
        )
        power_current = None
        for item in power_dirs:
            if int(item.get("rule_idx", -1)) == current_rule_idx:
                power_current = np.asarray(item.get("color_ordering", []), dtype=np.int64)
                break
        dist_decomp = _compute_area_puzzle_power_distances(
            area_series=area_series,
            num_colors=int(static.get("num_colors", 0)),
            power_ordering_current_rule=power_current,
            meta=meta,
        )

        # Page 1: left narrow reference/context + right dynamics panels.
        fig2 = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        gs2 = fig2.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 5.0],
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.12,
        )
        ax_ref = fig2.add_subplot(gs2[0, 0])
        ax_color = fig2.add_subplot(gs2[0, 1])
        ax_pg = fig2.add_subplot(gs2[1, 0])
        ax_dist = fig2.add_subplot(gs2[1, 1], sharex=ax_color)
        color_cols = sorted(
            [c for c in area_series.columns if c.startswith("area_color_")],
            key=lambda n: int(n.split("_")[-1]),
        )
        refs_panel = {
            "utilitarian": refs_area.get("dist_to_ref_utilitarian"),
            "nash": refs_area.get("dist_to_ref_nash"),
            "egalitarian": refs_area.get("dist_to_ref_egalitarian"),
            "rawlsian": refs_area.get("dist_to_ref_rawlsian"),
        }
        _draw_reference_optima_panel(ax=ax_ref, refs=refs_panel, num_colors=int(static.get("num_colors", 0)))

        for i, c in enumerate(color_cols):
            ax_color.plot(x, area_series[c].to_numpy(dtype=float), color=_sim_color(i), label=f"color_{i}")
        ax_color.set_title("Area Color Distribution Curves")
        ax_color.set_ylabel("share")
        _set_unit_ylim_visible(ax_color)
        if color_cols:
            ax_color.legend(loc="best", fontsize=8, ncol=min(4, len(color_cols)))

        _draw_area_personality_group_distribution(
            ax=ax_pg,
            pg_dist=pg_dist,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
        )

        # Background encodes elected ordering per step as stacked color bands
        # (top=rank 1 color ... bottom=last rank color).
        if "winning_option_id" in area_series.columns:
            ordering_bg = _build_elected_ordering_background_image(
                winning_option_ids=area_series["winning_option_id"].to_numpy(dtype=int),
                num_colors=int(static.get("num_colors", 0)),
            )
            if ordering_bg is not None:
                x0 = float(np.min(x)) - 0.5 if x.size > 0 else -0.5
                x1 = float(np.max(x)) + 0.5 if x.size > 0 else 0.5
                ax_dist.imshow(
                    ordering_bg,
                    origin="upper",
                    aspect="auto",
                    extent=[x0, x1, 0.0, 1.0],
                    interpolation="nearest",
                    zorder=0,
                )
        quality_mode = str(meta["run"].get("quality_target_mode", "reality"))
        q_source = quality_distance_source(quality_mode)
        ax_dist.plot(
            x,
            area_series["quality_distance"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.0,
            zorder=4,
            label=f"quality_distance ({q_source})",
        )
        if q_source == "puzzle_distance" and "dist_to_reality" in area_series.columns:
            ax_dist.plot(
                x,
                area_series["dist_to_reality"].to_numpy(dtype=float),
                color="tab:green",
                linestyle=":",
                linewidth=1.0,
                alpha=0.85,
                zorder=3,
                label="dist_to_reality",
            )
        ax_dist.set_title("quality_distance (mode-aware)")
        ax_dist.set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(ax_dist)
        ax_dist.legend(loc="best", fontsize=8)
        for a in (ax_color, ax_dist):
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig2.suptitle(suptitle, fontsize=11)
        if render_profile.area_core_page:
            pdf.savefig(fig2, dpi=140)
        plt.close(fig2)

        # Page 2: puzzle tracking (area-local puzzle distribution + puzzle distance).
        fig2p = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        gs2p = fig2p.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 5.0],
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.12,
        )
        ax_power = fig2p.add_subplot(gs2p[0, 0])
        ax_pcurve = fig2p.add_subplot(gs2p[0, 1], sharex=ax_color)
        ax_pg_p = fig2p.add_subplot(gs2p[1, 0])
        ax_pdist = fig2p.add_subplot(gs2p[1, 1], sharex=ax_pcurve)
        _draw_power_direction_panel(
            ax=ax_power,
            power_dirs=power_dirs,
            current_rule_idx=current_rule_idx,
            num_colors=int(static.get("num_colors", 0)),
        )

        puzzle_cols = sorted(
            [c for c in area_series.columns if c.startswith("puzzle_color_")],
            key=lambda n: int(n.split("_")[-1]),
        )
        if puzzle_cols:
            for i, c in enumerate(puzzle_cols):
                ax_pcurve.plot(x, area_series[c].to_numpy(dtype=float), color=_sim_color(i), label=f"color_{i}")
            ax_pcurve.set_title("Puzzle Distribution Curves")
            ax_pcurve.set_ylabel("share")
            _set_unit_ylim_visible(ax_pcurve)
            ax_pcurve.legend(loc="best", fontsize=8, ncol=min(4, len(puzzle_cols)))
        else:
            ax_pcurve.axis("off")
            ax_pcurve.set_title("Puzzle Distribution Curves")
            ax_pcurve.text(
                0.5,
                0.5,
                "Puzzle distribution not logged for this run.",
                ha="center",
                va="center",
                fontsize=10,
            )

        _draw_area_personality_group_distribution(
            ax=ax_pg_p,
            pg_dist=pg_dist,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
        )

        if "winning_option_id" in area_series.columns:
            ordering_bg = _build_elected_ordering_background_image(
                winning_option_ids=area_series["winning_option_id"].to_numpy(dtype=int),
                num_colors=int(static.get("num_colors", 0)),
            )
            if ordering_bg is not None:
                x0 = float(np.min(x)) - 0.5 if x.size > 0 else -0.5
                x1 = float(np.max(x)) + 0.5 if x.size > 0 else 0.5
                ax_pdist.imshow(
                    ordering_bg,
                    origin="upper",
                    aspect="auto",
                    extent=[x0, x1, 0.0, 1.0],
                    interpolation="nearest",
                    zorder=0,
                )
        ax_pdist.plot(
            x,
            area_series.get("puzzle_distance", pd.Series(np.nan, index=area_series.index)).to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="outcome↔puzzle",
            zorder=3,
        )
        if np.isfinite(dist_decomp["dist_outcome_power"]).any():
            ax_pdist.plot(
                x,
                dist_decomp["dist_outcome_power"].astype(float),
                color="tab:red",
                linewidth=1.3,
                label="outcome↔power",
                zorder=3,
            )
        if np.isfinite(puzzle_threshold):
            ax_pdist.axhline(
                puzzle_threshold,
                color="#4a4a4a",
                linestyle=":",
                linewidth=1.5,
                label="threshold",
                zorder=2,
            )
        ax_pdist.set_title("Puzzle Distance vs Outcome / Power")
        ax_pdist.set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(ax_pdist)
        if len(ax_pdist.lines) > 0:
            ax_pdist.legend(loc="upper right", fontsize=7, ncol=2)
        for a in (ax_pcurve, ax_pdist):
            if a.has_data():
                a.grid(True, alpha=0.25)
                a.set_xlabel("step")
        fig2p.suptitle(suptitle, fontsize=11)
        if render_profile.area_puzzle_page:
            pdf.savefig(fig2p, dpi=140)
        plt.close(fig2p)

        # Page 2b: use the newer puzzle/power decomposition page (formerly rendered later),
        # replacing the legacy split-decomposition page to avoid duplicate content.
        if render_profile.area_puzzle_page:
            _render_area_puzzle_gate_page(
                pdf=pdf,
                area_series=area_series,
                dist_decomp=dist_decomp,
                suptitle=suptitle,
                include_overview=False,
                include_decomposition=True,
            )

        # Page 3: vote-mode alignment diagnostics (with support/coverage context).
        # Render now, append as last page later.
        deferred_vote_mode_alignment_fig = None
        fig2m, axes2m = plt.subplots(
            3,
            1,
            figsize=(11.69, 8.27),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 0.75]},
        )
        axm = np.asarray(axes2m).ravel()
        has_mode_alignment = False
        has_mode_coverage = False
        for plot_idx, series_spec, title in (
            (
                0,
                (
                    ("altruistic_rank1_match_puzzle_share", "altruistic_votes_count", "tab:green", "-", "altruistic -> puzzle"),
                    ("self_regarding_rank1_match_puzzle_share", "self_regarding_votes_count", "tab:red", "--", "self-regarding -> puzzle"),
                ),
                "Rank-1 Match to Puzzle by Vote Mode [% of mode votes]",
            ),
            (
                1,
                (
                    ("altruistic_rank1_match_outcome_share", "altruistic_votes_count", "tab:green", "-", "altruistic -> elected"),
                    ("self_regarding_rank1_match_outcome_share", "self_regarding_votes_count", "tab:red", "--", "self-regarding -> elected"),
                ),
                "Rank-1 Match to Elected Outcome by Vote Mode [% of mode votes]",
            ),
        ):
            for value_col, count_col, color, ls, label in series_spec:
                if value_col not in area_series.columns:
                    continue
                y = area_series[value_col].to_numpy(dtype=float)
                if not np.isfinite(y).any():
                    continue
                axm[plot_idx].plot(x, y, color=color, linestyle=ls, linewidth=0.9, alpha=0.25)
                axm[plot_idx].plot(
                    x,
                    _rolling_mean_nan(y),
                    color=color,
                    linestyle=ls,
                    linewidth=1.6,
                    alpha=0.98,
                    label=label,
                )
                if count_col in area_series.columns:
                    cvals = area_series[count_col].to_numpy(dtype=float)
                    low_support = np.isfinite(cvals) & (cvals < float(_MODE_ALIGNMENT_LOW_SUPPORT_VOTES))
                    if np.any(low_support):
                        axm[plot_idx].fill_between(
                            x,
                            0.0,
                            100.0,
                            where=low_support,
                            color=color,
                            alpha=0.06,
                            linewidth=0.0,
                        )
                has_mode_alignment = True
            axm[plot_idx].set_title(title)
            axm[plot_idx].set_ylabel("%")
            _set_percent_ylim_visible(axm[plot_idx])
            if len(axm[plot_idx].lines) > 0:
                axm[plot_idx].legend(loc="best", fontsize=8)

        axm_cov = axm[2]
        axm_cov.set_title("Vote-mode Coverage (solid=mode share, black=count)")
        axm_cov.set_ylabel("% of votes")
        for col, color, label in (
            ("altruistic_vote_share", "tab:green", "altruistic vote share"),
            ("self_regarding_vote_share", "tab:red", "self-regarding vote share"),
        ):
            if col not in area_series.columns:
                continue
            y = area_series[col].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            axm_cov.plot(x, y, color=color, linewidth=0.9, alpha=0.25)
            axm_cov.plot(x, _rolling_mean_nan(y), color=color, linewidth=1.5, alpha=0.95, label=label)
            has_mode_coverage = True
        _set_percent_ylim_visible(axm_cov)

        cov_rhs = None
        if "vote_count_total" in area_series.columns:
            votes_total = area_series["vote_count_total"].to_numpy(dtype=float)
            if np.isfinite(votes_total).any():
                cov_rhs = axm_cov.twinx()
                cov_rhs.plot(x, votes_total, color="black", linewidth=1.1, alpha=0.85, label="mode vote count")
                cov_rhs.axhline(
                    float(_MODE_ALIGNMENT_LOW_SUPPORT_VOTES),
                    color="black",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )
                cov_rhs.set_ylabel("count", color="black")
                cov_rhs.tick_params(axis="y", colors="black")
                has_mode_coverage = True

        if len(axm_cov.lines) > 0 or (cov_rhs is not None and len(cov_rhs.lines) > 0):
            h1, l1 = axm_cov.get_legend_handles_labels()
            if cov_rhs is not None:
                h2, l2 = cov_rhs.get_legend_handles_labels()
                axm_cov.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            else:
                axm_cov.legend(loc="best", fontsize=8)

        if not has_mode_alignment and not has_mode_coverage:
            for a in axm:
                a.text(0.5, 0.5, "Mode alignment unavailable (requires votes + puzzle logging)", ha="center", va="center")
                a.set_yticks([])
        for a in axm:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig2m.suptitle(suptitle, fontsize=11)
        fig2m.tight_layout()
        deferred_vote_mode_alignment_fig = fig2m

        # Page 4: group puzzle opportunity alignment + compact divergence diagnostics.
        fig2g, axes2g = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        axg = np.asarray(axes2g).ravel()
        opp_df = _compute_group_puzzle_opportunity_distances(
            area_series=area_series,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
            meta=meta,
        )
        groups_sorted = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist()) if not area_group_series.empty else []
        group_resident_count_static: dict[int, float] = {}
        if not area_group_series.empty and groups_sorted:
            for g in groups_sorted:
                vals = area_group_series.loc[area_group_series["group_idx"] == g, "residents"].to_numpy(dtype=float)
                group_resident_count_static[int(g)] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

        def _opp_group_is_small(g: int) -> bool:
            cnt = group_resident_count_static.get(int(g), float("nan"))
            return bool(np.isfinite(cnt) and cnt < float(_SMALL_GROUP_MIN_RESIDENTS))

        def _opp_group_alpha(g: int, *, normal: float = 0.95, small: float = 0.14) -> float:
            return float(small if _opp_group_is_small(int(g)) else normal)

        has_small_opp_groups = any(_opp_group_is_small(int(g)) for g in groups_sorted)
        major_groups_sorted = [int(g) for g in groups_sorted if not _opp_group_is_small(int(g))]
        spread_groups_sorted = major_groups_sorted if major_groups_sorted else list(groups_sorted)

        has_opp = False
        if not opp_df.empty:
            xs = opp_df["step"].to_numpy(dtype=float)
            shown_opp_groups = 0
            for g in groups_sorted:
                c = f"group_{g}_puzzle_opp_dist"
                if c in opp_df.columns:
                    y = opp_df[c].to_numpy(dtype=float)
                    if np.isfinite(y).any():
                        color = get_group_color(int(g))
                        is_small = _opp_group_is_small(int(g))
                        axg[0].plot(
                            xs,
                            y,
                            color=color,
                            linewidth=0.8 if is_small else 0.9,
                            alpha=_opp_group_alpha(int(g), normal=0.22, small=0.06),
                        )
                        axg[0].plot(
                            xs,
                            _rolling_mean_nan(y),
                            color=color,
                            linewidth=1.3 if is_small else 1.5,
                            alpha=_opp_group_alpha(int(g), normal=0.98, small=0.18),
                            label=f"g{g}" if not is_small else "_nolegend_",
                        )
                        if not is_small:
                            shown_opp_groups += 1
                        has_opp = True
        axg[0].set_title("Group Opportunity Alignment to Puzzle [distance(group ordering, puzzle ordering)]")
        axg[0].set_ylabel("distance [0..1]\n(lower=more aligned)")
        _set_unit_ylim_visible(axg[0])
        if not opp_df.empty and shown_opp_groups > 0:
            axg[0].legend(loc="best", fontsize=8, ncol=min(5, len(axg[0].lines)))
        if has_small_opp_groups:
            axg[0].text(
                0.01,
                0.03,
                f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
                transform=axg[0].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )

        has_behavior = False
        opp_rhs = None
        if not area_group_series.empty and groups_sorted:
            steps_g = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
            part_vals_all = area_group_series["participants"].to_numpy(dtype=float)
            non_alt_vals_all = area_group_series["self_regarding_voters"].to_numpy(dtype=float)
            non_alt_share_vals = np.full(len(area_group_series), np.nan, dtype=np.float32)
            np.divide(
                100.0 * non_alt_vals_all,
                part_vals_all,
                out=non_alt_share_vals,
                where=part_vals_all > 0.0,
            )
            p_non_alt_share = (
                area_group_series.assign(
                    non_alt_share=non_alt_share_vals
                )
                .pivot(index="step", columns="group_idx", values="non_alt_share")
                .reindex(index=steps_g.astype(int), columns=spread_groups_sorted)
                .astype(float)
            )
            p_turn = (
                area_group_series.pivot(index="step", columns="group_idx", values="turnout")
                .reindex(index=steps_g.astype(int), columns=spread_groups_sorted)
                .astype(float)
            )
            non_alt_mat = p_non_alt_share.to_numpy(dtype=float)
            turnout_mat = p_turn.to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                non_alt_range = np.nanmax(non_alt_mat, axis=1) - np.nanmin(non_alt_mat, axis=1)
                turnout_range = np.nanmax(turnout_mat, axis=1) - np.nanmin(turnout_mat, axis=1)
            non_alt_all_nan = np.all(~np.isfinite(non_alt_mat), axis=1)
            turnout_all_nan = np.all(~np.isfinite(turnout_mat), axis=1)
            non_alt_range[non_alt_all_nan] = np.nan
            turnout_range[turnout_all_nan] = np.nan
            if np.isfinite(non_alt_range).any():
                axg[1].plot(steps_g, non_alt_range, color="tab:red", linewidth=0.9, alpha=0.25, linestyle=":")
                axg[1].plot(
                    steps_g,
                    _rolling_mean_nan(non_alt_range),
                    color="tab:red",
                    linewidth=1.6,
                    alpha=0.95,
                    linestyle=":",
                    label="non-alt share range across groups",
                )
                has_behavior = True
            if np.isfinite(turnout_range).any():
                axg[1].plot(steps_g, turnout_range, color="tab:blue", linewidth=0.9, alpha=0.25, linestyle="-")
                axg[1].plot(
                    steps_g,
                    _rolling_mean_nan(turnout_range),
                    color="tab:blue",
                    linewidth=1.6,
                    alpha=0.95,
                    linestyle="-",
                    label="turnout range across groups",
                )
                has_behavior = True
        if not opp_df.empty:
            opp_cols = [f"group_{int(g)}_puzzle_opp_dist" for g in spread_groups_sorted if f"group_{int(g)}_puzzle_opp_dist" in opp_df.columns]
            if opp_cols:
                opp_mat = opp_df[opp_cols].to_numpy(dtype=float)
                with np.errstate(invalid="ignore"):
                    opp_mean = np.nanmean(opp_mat, axis=1)
                    opp_range = np.nanmax(opp_mat, axis=1) - np.nanmin(opp_mat, axis=1)
                opp_all_nan = np.all(~np.isfinite(opp_mat), axis=1)
                opp_mean[opp_all_nan] = np.nan
                opp_range[opp_all_nan] = np.nan
                if np.isfinite(opp_mean).any() or np.isfinite(opp_range).any():
                    opp_rhs = axg[1].twinx()
                    if np.isfinite(opp_mean).any():
                        opp_rhs.plot(xs, _rolling_mean_nan(opp_mean), color="tab:green", linewidth=1.35, alpha=0.95, label="mean opp. dist")
                    if np.isfinite(opp_range).any():
                        opp_rhs.plot(xs, _rolling_mean_nan(opp_range), color="tab:purple", linewidth=1.25, alpha=0.95, linestyle="--", label="opp. dist range")
                    _set_unit_ylim_visible(opp_rhs)
                    opp_rhs.set_ylabel("puzzle-opportunity distance [0..1]", color="tab:green")
                    opp_rhs.tick_params(axis="y", colors="tab:green")
        axg[1].set_title("Cross-group Spread Diagnostics (behavior divergence + opportunity spread)")
        axg[1].set_ylabel("%")
        _set_percent_ylim_visible(axg[1])
        if len(axg[1].lines) > 0 or (opp_rhs is not None and len(opp_rhs.lines) > 0):
            h1, l1 = axg[1].get_legend_handles_labels()
            if opp_rhs is not None:
                h2, l2 = opp_rhs.get_legend_handles_labels()
                axg[1].legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=2)
            else:
                axg[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=2)
        if has_small_opp_groups:
            if major_groups_sorted:
                note = f"Spread excludes groups with residents < {_SMALL_GROUP_MIN_RESIDENTS}."
            else:
                note = f"All groups are < {_SMALL_GROUP_MIN_RESIDENTS}; spread uses all groups."
            axg[1].text(
                0.01,
                0.03,
                note,
                transform=axg[1].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
        if not has_opp:
            axg[0].text(0.5, 0.5, "Opportunity alignment unavailable (requires puzzle distribution logging)", ha="center", va="center")
            axg[0].set_yticks([])
        if not has_behavior:
            axg[1].text(0.5, 0.5, "Group divergence series unavailable", ha="center", va="center")
            axg[1].set_yticks([])
        for a in axg:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig2g.suptitle(suptitle, fontsize=11)
        fig2g.tight_layout()
        if render_profile.area_group_opportunity_page:
            pdf.savefig(fig2g, dpi=140)
        plt.close(fig2g)

        if render_profile.area_puzzle_gate_page:
            _render_area_puzzle_gate_page(
                pdf=pdf,
                area_series=area_series,
                dist_decomp=dist_decomp,
                suptitle=suptitle,
                include_overview=True,
                include_decomposition=False,
            )

        # Following pages (group diagnostics etc.) come after the core area + puzzle analysis pages.
        if render_profile.area_group_diagnostics_pages and (not area_group_series.empty):
            _render_area_group_pages(
                pdf=pdf,
                area_group_series=area_group_series,
                x=x,
                turnout=turnout,
                participants=participants,
                eligible=eligible,
                suptitle=suptitle,
            )
        # Summary packet trim: learning-causal page is intentionally disabled.

        # Page 10: gini assets + assets share by group
        fig1, axes1 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        ax1 = np.asarray(axes1).ravel()
        ax1[0].plot(x, area_series["gini_assets"].to_numpy(dtype=float), color="tab:red")
        ax1[0].set_title("Gini Assets [0..100]")
        ax1[0].set_ylabel("gini")
        _set_percent_ylim_visible(ax1[0])
        assets_share_payload = _prepare_group_assets_share_series(area_group_series=area_group_series)
        if assets_share_payload is not None:
            steps_assets, groups_assets, p_assets_share = assets_share_payload
            for g in groups_assets:
                ax1[1].plot(
                    steps_assets,
                    p_assets_share[g].to_numpy(dtype=float),
                    color=get_group_color(int(g)),
                    linewidth=1.8,
                    label=f"g{g}",
                )
            ax1[1].set_title("Assets share by Group")
            ax1[1].set_ylabel("share")
            _set_unit_ylim_visible(ax1[1])
            if groups_assets:
                ax1[1].legend(loc="best", fontsize=8, ncol=min(5, len(groups_assets)))
        else:
            ax1[1].text(0.5, 0.5, "Assets share by group unavailable", ha="center", va="center")
            ax1[1].set_yticks([])

        for a in ax1:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig1.suptitle(suptitle, fontsize=11)
        fig1.tight_layout()
        if render_profile.area_assets_page:
            pdf.savefig(fig1, dpi=140)
        plt.close(fig1)

        # Then the rest: first group means page, then dist_to_ref + area means.
        if render_profile.area_group_means_page and (not area_group_series.empty):
            _render_area_group_means_page(
                pdf=pdf,
                area_group_series=area_group_series,
                x=x,
                gini_dissatisfaction=area_series["gini_dissatisfaction"].to_numpy(dtype=float),
            )

        fig3, axes3 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        ax3 = np.asarray(axes3).ravel()
        for col, color, label in (
            ("dist_to_ref_utilitarian", "tab:blue", "utilitarian"),
            ("dist_to_ref_nash", "tab:purple", "nash"),
            ("dist_to_ref_egalitarian", "tab:orange", "egalitarian"),
            ("dist_to_ref_rawlsian", "tab:red", "rawlsian"),
        ):
            if col in area_series.columns:
                vals = area_series[col].to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    ax3[0].plot(x, vals, color=color, label=label)
        ax3[0].set_title("dist_to_ref_*")
        ax3[0].set_ylabel("distance [0..1] (lower better)")
        _set_unit_ylim_visible(ax3[0])
        if len(ax3[0].lines) > 0:
            ax3[0].legend(loc="best", fontsize=8)

        area_means = _compute_area_weighted_means_from_group_series(area_group_series=area_group_series)
        if area_means is not None:
            ax3[1].plot(
                area_means["step"].to_numpy(dtype=float),
                area_means["mean_assets"].to_numpy(dtype=float),
                color="tab:blue",
                linewidth=1.6,
                label="mean_assets",
            )
            ax3b = ax3[1].twinx()
            ax3b.plot(
                area_means["step"].to_numpy(dtype=float),
                area_means["mean_dissatisfaction"].to_numpy(dtype=float),
                color="tab:orange",
                linestyle="--",
                linewidth=1.6,
                label="mean_dissatisfaction",
            )
            ax3[1].set_ylabel("assets", color="tab:blue")
            ax3b.set_ylabel("dissatisfaction [0..1]", color="tab:orange")
            _set_unit_ylim_visible(ax3b)
            ax3[1].tick_params(axis="y", colors="tab:blue")
            ax3b.tick_params(axis="y", colors="tab:orange")
            h1, l1 = ax3[1].get_legend_handles_labels()
            h2, l2 = ax3b.get_legend_handles_labels()
            if h1 or h2:
                ax3[1].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        else:
            ax3[1].text(0.5, 0.5, "Area mean assets/dissatisfaction unavailable", ha="center", va="center")
            ax3[1].set_yticks([])
        ax3[1].set_title("Area Mean Assets + Mean Dissatisfaction")
        for a in ax3:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig3.suptitle(suptitle, fontsize=11)
        fig3.tight_layout()
        if render_profile.area_dist_to_ref_page:
            pdf.savefig(fig3, dpi=140)
        plt.close(fig3)

        # Keep this diagnostics page as the final page in the area PDF packet.
        if deferred_vote_mode_alignment_fig is not None:
            if render_profile.area_vote_mode_alignment_page:
                pdf.savefig(deferred_vote_mode_alignment_fig, dpi=140)
            plt.close(deferred_vote_mode_alignment_fig)

def _render_area_group_pages(
    *,
    pdf: PdfPages,
    area_group_series: pd.DataFrame,
    x: np.ndarray,
    turnout: np.ndarray,
    participants: np.ndarray,
    eligible: np.ndarray,
    suptitle: str,
) -> None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return

    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    pivot_optional = lambda col: pivot(col) if col in area_group_series.columns else None
    p_res = pivot("residents")
    p_elig = pivot("eligible")
    p_part = pivot("participants")
    p_non_alt = pivot("self_regarding_voters")
    p_turn = pivot("turnout")
    p_assets = pivot("mean_assets")
    p_dissat = pivot("mean_dissatisfaction")
    p_res_share = pivot("resident_share")
    p_part_share = pivot("participant_share")
    p_part_delta = pivot_optional("participants_mean_delta_rel")
    p_abs_delta = pivot_optional("abstainers_mean_delta_rel")
    p_part_fee = pivot_optional("participants_mean_fee")
    p_part_fee_assets = pivot_optional("participants_mean_fee_over_assets")
    p_q_std_within = pivot_optional("group_std_q_participation")
    p_p_std_within = pivot_optional("group_std_participation_probability")
    p_gini_assets_within = pivot_optional("group_gini_assets_within")
    p_gini_diss_within = pivot_optional("group_gini_dissatisfaction_within")
    p_alt_a_update = pivot_optional("altruistic_voters_mean_altruism_update_proxy")
    p_non_alt_a_update = pivot_optional("self_regarding_voters_mean_altruism_update_proxy")
    p_alt_a_delta_exact = pivot_optional("altruistic_voters_mean_altruism_delta")
    p_non_alt_a_delta_exact = pivot_optional("self_regarding_voters_mean_altruism_delta")
    p_mode_switch = pivot_optional("vote_mode_switch_share")
    p_mode_switch_from_alt = pivot_optional("vote_mode_switch_from_altruistic_share")
    p_mode_switch_from_non_alt = pivot_optional("vote_mode_switch_from_self_regarding_share")
    p_part_to_abs_switch = pivot_optional("participation_switch_to_abstain_share")
    p_alt_dsig = pivot_optional("altruistic_voters_mean_dissatisfaction_signal")
    p_non_alt_dsig = pivot_optional("self_regarding_voters_mean_dissatisfaction_signal")

    # Page 2: top participants/eligible/self-regarding; bottom composition share, both with right-side references.
    fig4 = plt.figure(figsize=(11.69, 8.27))
    gs4 = fig4.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    top = gs4[0].subgridspec(1, 2, width_ratios=[4.0, 0.15], wspace=0.01)
    bottom = gs4[1].subgridspec(1, 2, width_ratios=[4.0, 0.15], wspace=0.01)
    ax4_left = fig4.add_subplot(top[0, 0])
    ax4_ref = fig4.add_subplot(top[0, 1])
    ax4_bottom = fig4.add_subplot(bottom[0, 0])
    ax4_bottom_ref = fig4.add_subplot(bottom[0, 1])
    resident_share_static_vals: list[float] = []
    for g in groups:
        vals = area_group_series[area_group_series["group_idx"] == g]["resident_share"].dropna()
        resident_share_static_vals.append(float(vals.iloc[0]) if not vals.empty else 0.0)

    resident_share_static = np.asarray(resident_share_static_vals, dtype=float)
    group_resident_count_static: dict[int, float] = {}
    if groups:
        for g in groups:
            vals = p_res[g].to_numpy(dtype=float)
            group_resident_count_static[int(g)] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

    def _group_is_small(g: int) -> bool:
        cnt = group_resident_count_static.get(int(g), float("nan"))
        return bool(np.isfinite(cnt) and cnt < float(_SMALL_GROUP_MIN_RESIDENTS))

    def _group_alpha(g: int, *, normal: float = 0.9, small: float = 0.14) -> float:
        return float(small if _group_is_small(int(g)) else normal)

    has_small_groups = any(_group_is_small(int(g)) for g in groups)

    ax4_ref.set_xlim(float(np.min(steps)) if steps.size > 0 else 0.0, float(np.max(steps)) if steps.size > 0 else 1.0)
    max_res = 0.0
    if groups:
        max_res = float(np.nanmax([np.nanmax(p_res[g].to_numpy(dtype=float)) for g in groups]))
    ax4_ref.set_ylim(0.0, max(1.0, max_res * 1.05))
    ax4_ref.set_title("Total\nCount")
    ax4_ref.axis("off")

    for g in groups:
        color = get_group_color(int(g))
        ax4_left.plot(steps, p_part[g].to_numpy(dtype=float), color=color, linewidth=1.15, label=f"g{g} participants")
        ax4_left.plot(steps, p_elig[g].to_numpy(dtype=float), color=color, linestyle=":", alpha=0.85, linewidth=1.2)
    ax4_left.set_title("Participants (solid) + Eligible (dotted) by Group")
    ax4_left.set_ylabel("count")
    ax4_left.grid(True, alpha=0.25)
    ax4_left.set_xlabel("step")
    if groups:
        ax4_left.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8, ncol=min(5, len(groups)))
    ax4_left.set_ylim(bottom=0.0)
    ax4_left.margins(x=0.0, y=0.0)
    ax4_left.spines["bottom"].set_position(("data", 0.0))

    # Top reference: thin dashed residents-by-group traces (no axis).
    for g in groups:
        ax4_ref.plot(
            steps,
            p_res[g].to_numpy(dtype=float),
            color=get_group_color(int(g)),
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )
    left_stack = [p_part_share[g].to_numpy(dtype=float) for g in groups]
    if left_stack:
        ax4_bottom.stackplot(
            steps,
            *left_stack,
            labels=[f"g{g}" for g in groups],
            colors=[get_group_color(int(g)) for g in groups],
            alpha=0.9,
        )
    ax4_bottom.set_title("Participant Composition Share by Group")
    ax4_bottom.set_ylabel("share")
    ax4_bottom.set_ylim(0.0, 1.0)
    ax4_bottom.grid(True, alpha=0.25)
    ax4_bottom.set_xlabel("step")
    ax4_bottom.set_ylim(bottom=0.0)
    ax4_bottom.margins(x=0.0, y=0.0)
    ax4_bottom.spines["bottom"].set_position(("data", 0.0))

    ax4_bottom_ref.set_ylim(0.0, 1.0)
    ax4_bottom_ref.set_title("Total\nShare")
    ax4_bottom_ref.axis("off")
    bottom_share = 0.0
    for g, s in zip(groups, resident_share_static):
        ax4_bottom_ref.bar(0, s, bottom=bottom_share, width=0.2, color=get_group_color(int(g)), edgecolor="none")
        y_mid = bottom_share + (float(s) / 2.0)
        label = f"g{int(g)}"
        if float(s) >= 0.10:
            label = f"g{int(g)}\n{int(round(float(s) * 100.0))}%"
        r, gg, b, _ = to_rgba(get_group_color(int(g)))
        luminance = 0.299 * r + 0.587 * gg + 0.114 * b
        txt_color = "black" if luminance > 0.55 else "white"
        if float(s) >= 0.045:
            ax4_bottom_ref.text(
                0.0,
                y_mid,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
                fontweight="bold",
            )
        bottom_share += s

    fig4.tight_layout()
    pdf.savefig(fig4, dpi=140)
    plt.close(fig4)

    # Page 3: self-regarding share among participants by group (top), Turnout by Group (bottom).
    fig3, ax3 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax3[0].set_title("Self-regarding Share Among Participants by Group")
    ax3[0].set_ylabel("%")
    for g in groups:
        color = get_group_color(int(g))
        part_vals = p_part[g].to_numpy(dtype=float)
        non_alt_vals = p_non_alt[g].to_numpy(dtype=float)
        share = np.zeros_like(part_vals, dtype=float)
        np.divide(non_alt_vals, part_vals, out=share, where=part_vals > 0.0)
        y = share * 100.0
        ax3[0].plot(
            steps,
            y,
            color=color,
            linestyle="-",
            linewidth=0.8,
            alpha=0.24,
        )
        ax3[0].plot(
            steps,
            _rolling_mean_nan(y),
            color=color,
            linestyle="-",
            linewidth=1.55,
            alpha=0.98,
            label=f"g{g}",
        )
    total_participants = p_part.sum(axis=1).to_numpy(dtype=float)
    total_non_alt = p_non_alt.sum(axis=1).to_numpy(dtype=float)
    total_share = np.zeros_like(total_non_alt, dtype=float)
    np.divide(total_non_alt, total_participants, out=total_share, where=total_participants > 0.0)
    ax3[0].plot(
        steps,
        total_share * 100.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.28,
    )
    ax3[0].plot(
        steps,
        _rolling_mean_nan(total_share * 100.0),
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.98,
        label="total self-regarding share",
    )
    _set_percent_ylim_visible(ax3[0])
    if groups:
        ax3[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups) + 1))
    for g in groups:
        ax3[1].plot(steps, p_turn[g].to_numpy(dtype=float), color=get_group_color(int(g)), linewidth=0.9, label=f"g{g}")
    ax3[1].plot(x, turnout, color="black", linestyle="--", linewidth=1.4, alpha=0.9, label="turnout total")
    ax3[1].set_title("Turnout by Group [% of Residents]")
    ax3[1].set_ylabel("%")
    _set_percent_ylim_visible(ax3[1])
    for a in ax3:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig3.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig3.tight_layout()
    pdf.savefig(fig3, dpi=140)
    plt.close(fig3)

    # Page 4: split altruistic/self-regarding diagnostics into three readable panels.
    fig5, ax5 = plt.subplots(3, 1, figsize=(11.69, 8.27), sharex=True)

    def _set_percent_ylim_adaptive(ax, series_list: list[np.ndarray]) -> None:
        finite_chunks = []
        for s in series_list:
            arr = np.asarray(s, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                finite_chunks.append(arr)
        if not finite_chunks:
            _set_percent_ylim_visible(ax)
            return
        vals = np.concatenate(finite_chunks)
        ymax = float(np.nanmax(vals))
        if (not np.isfinite(ymax)) or ymax <= 0.0:
            top = 10.0
        else:
            top = min(100.0, max(10.0, ymax * 1.10))
        ax.set_ylim(-float(_Y_PAD_PERCENT), float(top) + float(_Y_PAD_PERCENT))
    total_participants = p_part.sum(axis=1).to_numpy(dtype=float)
    total_non_alt = p_non_alt.sum(axis=1).to_numpy(dtype=float)
    total_alt = np.maximum(0.0, total_participants - total_non_alt)
    total_residents = p_res.sum(axis=1).to_numpy(dtype=float)
    finite_res = total_residents[np.isfinite(total_residents) & (total_residents > 0.0)]
    denom_agents = float(finite_res[0]) if finite_res.size > 0 else float("nan")

    turnout_pct = np.zeros_like(total_participants, dtype=float)
    non_alt_pct = np.zeros_like(total_non_alt, dtype=float)
    alt_pct = np.zeros_like(total_alt, dtype=float)
    if np.isfinite(denom_agents) and denom_agents > 0.0:
        turnout_pct = 100.0 * (total_participants / denom_agents)
        non_alt_pct = 100.0 * (total_non_alt / denom_agents)
        alt_pct = 100.0 * (total_alt / denom_agents)

    ax5[0].set_title("Total Turnout / Altruistic / Self-regarding Voters [% of Area Agents]")
    ax5[0].set_ylabel("%")
    ax5[0].plot(steps, turnout_pct, color="black", linewidth=1.5, alpha=0.95, label="total turnout")
    ax5[0].plot(steps, alt_pct, color="tab:green", linewidth=1.35, alpha=0.95, label="total altruistic")
    ax5[0].plot(steps, non_alt_pct, color="tab:red", linewidth=1.35, alpha=0.95, label="total self-regarding")
    _set_percent_ylim_adaptive(ax5[0], [turnout_pct, alt_pct, non_alt_pct])
    ax5[0].legend(loc="best", fontsize=8)

    ax5[1].set_title("Self-regarding Votes by Group + Total Altruistic Votes [% of Area Agents]")
    ax5[1].set_ylabel("%")
    non_alt_group_series: list[np.ndarray] = []
    for g in groups:
        non_alt_vals = p_non_alt[g].to_numpy(dtype=float)
        group_non_alt_pct = np.zeros_like(non_alt_vals, dtype=float)
        if np.isfinite(denom_agents) and denom_agents > 0.0:
            group_non_alt_pct = 100.0 * (non_alt_vals / denom_agents)
        non_alt_group_series.append(group_non_alt_pct)
        ax5[1].plot(
            steps,
            group_non_alt_pct,
            color=get_group_color(int(g)),
            linewidth=1.15,
            alpha=0.92,
            label=f"g{g} self-regarding",
        )
    ax5[1].plot(
        steps,
        alt_pct,
        color="black",
        linestyle="--",
        linewidth=1.35,
        alpha=0.95,
        label="total altruistic",
    )
    _set_percent_ylim_adaptive(ax5[1], non_alt_group_series + [alt_pct])
    if groups:
        ax5[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups) + 1))

    ax5[2].set_title("Altruistic Votes by Group [% of Area Agents]")
    ax5[2].set_ylabel("%")
    alt_group_series: list[np.ndarray] = []
    for g in groups:
        part_vals = p_part[g].to_numpy(dtype=float)
        non_alt_vals = p_non_alt[g].to_numpy(dtype=float)
        alt_vals = np.maximum(0.0, part_vals - non_alt_vals)
        group_alt_pct = np.zeros_like(alt_vals, dtype=float)
        if np.isfinite(denom_agents) and denom_agents > 0.0:
            group_alt_pct = 100.0 * (alt_vals / denom_agents)
        alt_group_series.append(group_alt_pct)
        ax5[2].plot(
            steps,
            group_alt_pct,
            color=get_group_color(int(g)),
            linewidth=1.15,
            alpha=0.92,
            label=f"g{g} altruistic",
        )
    _set_percent_ylim_adaptive(ax5[2], alt_group_series)
    if groups:
        ax5[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))

    for a in ax5:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig5.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig5.tight_layout()
    pdf.savefig(fig5, dpi=140)
    plt.close(fig5)

    # Summary packet trim: pages after this point are intentionally disabled.
    # This keeps only the first three group-diagnostics pages in the area PDF.
    return

def _render_area_group_means_page(
    *,
    pdf: PdfPages,
    area_group_series: pd.DataFrame,
    x: np.ndarray,
    gini_dissatisfaction: np.ndarray,
) -> None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return
    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    p_dissat = pivot("mean_dissatisfaction")

    fig, ax = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax[0].plot(x, gini_dissatisfaction, color="tab:purple", linewidth=1.8)
    ax[0].set_title("Gini Dissatisfaction [0..100]")
    ax[0].set_ylabel("gini")
    _set_percent_ylim_visible(ax[0])
    for g in groups:
        color = get_group_color(int(g))
        ax[1].plot(steps, p_dissat[g].to_numpy(dtype=float), color=color, linewidth=1.8, label=f"g{g}")
    ax[1].set_title("Mean Dissatisfaction by Group")
    ax[1].set_ylabel("dissatisfaction [0..1]")
    _set_unit_ylim_visible(ax[1])
    for a in ax:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    if groups:
        ax[1].legend(loc="best", fontsize=8, ncol=min(5, len(groups)))
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)

def _prepare_group_assets_share_series(
    *,
    area_group_series: pd.DataFrame,
) -> tuple[np.ndarray, list[int], pd.DataFrame] | None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return None
    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    p_res = pivot("residents")
    p_assets = pivot("mean_assets")
    p_group_assets = p_assets * p_res
    asset_totals = p_group_assets.sum(axis=1).to_numpy(dtype=float)
    p_assets_share = p_group_assets.copy()
    for g in groups:
        vals = p_group_assets[g].to_numpy(dtype=float)
        share = np.zeros_like(vals, dtype=float)
        np.divide(vals, asset_totals, out=share, where=asset_totals > 0.0)
        p_assets_share[g] = share
    return steps, groups, p_assets_share

def _compute_area_weighted_means_from_group_series(*, area_group_series: pd.DataFrame) -> pd.DataFrame | None:
    if area_group_series.empty:
        return None
    required = {"step", "residents", "mean_assets", "mean_dissatisfaction"}
    if not required.issubset(area_group_series.columns):
        return None
    rows: list[dict[str, float]] = []
    for step, block in area_group_series.groupby("step", sort=True):
        w = block["residents"].to_numpy(dtype=float)
        if w.size == 0 or float(np.sum(w)) <= 0.0:
            continue
        assets = block["mean_assets"].to_numpy(dtype=float)
        dissat = block["mean_dissatisfaction"].to_numpy(dtype=float)
        rows.append(
            {
                "step": float(step),
                "mean_assets": float(np.average(assets, weights=w)),
                "mean_dissatisfaction": float(np.average(dissat, weights=w)),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

def _draw_reference_optima_panel(*, ax, refs: dict[str, np.ndarray | None], num_colors: int) -> None:
    ax.set_title("Fixed Reference Optima")
    names = ("utilitarian", "nash", "egalitarian", "rawlsian")
    x_pos = np.arange(len(names), dtype=float)
    any_valid = False

    for xi, name in enumerate(names):
        ref = refs.get(name)
        if ref is None or ref.size != int(num_colors):
            continue
        for color_id in range(int(num_colors)):
            x_val = float(x_pos[xi])
            y_val = float(ref[color_id])
            # short horizontal segment centered on dot to separate close values
            ax.plot(
                [x_val - 0.185, x_val + 0.185],
                [y_val, y_val],
                color=_sim_color(color_id),
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                [x_val],
                [y_val],
                s=34,
                color=_sim_color(color_id),
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
            )
            any_valid = True

    if not any_valid:
        ax.text(0.5, 0.5, "reference distributions unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.set_xlim(-0.5, len(names) - 0.5)
    _set_unit_ylim_visible(ax)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["util", "nash", "egal", "rawl"], fontsize=8)
    ax.set_ylabel("share")
    ax.grid(True, axis="y", alpha=0.25)

def _draw_area_personality_group_distribution(
    *,
    ax,
    pg_dist: np.ndarray,
    personality_groups: np.ndarray,
    num_colors: int,
) -> None:
    ax.set_title("Personality Group Dists")
    dist = np.asarray(pg_dist, dtype=float).reshape(-1)
    if dist.size == 0 or not np.isfinite(dist).any():
        ax.axis("off")
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        return
    n_groups = int(dist.size)
    x = np.arange(n_groups, dtype=float)
    # Background: show each group's preference ordering as stacked color stripes.
    if personality_groups.ndim == 2 and personality_groups.shape[0] >= n_groups and int(num_colors) > 0:
        n_slots = int(min(int(num_colors), personality_groups.shape[1]))
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
    # Foreground bars: keep black-frame histogram look, add subtle group color fill.
    bars = ax.bar(
        x,
        dist,
        width=0.75,
        facecolor="none",
        edgecolor=[get_group_color(i) for i in range(n_groups)],
        linewidth=1.0,
        zorder=2,
    )
    _set_unit_ylim_visible(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"g{i}" for i in range(n_groups)], fontsize=7)
    ax.set_yticks([])
    ax.grid(True, axis="y", alpha=0.2)
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
                boxstyle="round,pad=0.18,rounding_size=0.08",
                alpha=0.95,
            )
        )
    for b in bars:
        h = float(b.get_height())
        gi = int(round(float(b.get_x() + b.get_width() / 2.0)))
        label_color = get_group_color(max(0, min(n_groups - 1, gi)))
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(0.98, h + 0.02),
            f"{100.0 * h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color=label_color,
            fontweight="bold",
        )

def _sim_color(color_idx: int) -> str:
    if 0 <= int(color_idx) < len(SIM_COLORS):
        name = SIM_COLORS[int(color_idx)]
        # Matplotlib normalizes both spellings; keep a single one for consistency.
        return "LightGrey" if name == "LightGray" else str(name)
    return "black"

def _compute_area_power_direction_orderings(
    *,
    area_group_series: pd.DataFrame,
    personality_groups: np.ndarray,
    num_colors: int,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Static self-regarding counterfactual outcomes per voting rule for this area."""
    if area_group_series.empty or personality_groups.ndim != 2 or num_colors <= 0:
        return []
    if "group_idx" not in area_group_series.columns or "residents" not in area_group_series.columns:
        return []
    step0 = int(area_group_series["step"].min()) if "step" in area_group_series.columns else 1
    block = area_group_series[area_group_series["step"].astype(int) == step0].copy()
    if block.empty:
        return []

    n_groups = int(personality_groups.shape[0])
    residents_by_group = np.zeros(n_groups, dtype=int)
    for _, row in block.iterrows():
        gi = int(row["group_idx"])
        if 0 <= gi < n_groups:
            residents_by_group[gi] = int(max(0, int(row.get("residents", 0))))
    if int(residents_by_group.sum()) <= 0:
        return []

    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)
    dist_func = _summary_ordering_distance_func(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))

    pref_rows: list[np.ndarray] = []
    for gi in range(min(n_groups, int(personality_groups.shape[0]))):
        cnt = int(residents_by_group[gi])
        if cnt <= 0:
            continue
        target = np.asarray(personality_groups[gi][:num_colors], dtype=np.int64)
        scores = score_options_c2(
            target_ordering=target,
            options=options,
            distance_func=dist_func,
            color_search_pairs=search_pairs,
        ).astype(np.float32)
        pref_rows.append(np.repeat(scores[None, :], cnt, axis=0))
    if not pref_rows:
        return []
    pref_table = np.vstack(pref_rows)

    # Keep this panel deterministic and interpretable: exclude Random baseline.
    rule_fns = [majority_rule, approval_voting, utilitarian_rule, borda_rule, schulze_rule]
    rule_names = ["Majority", "Approval", "Utilitarian", "Borda", "Schulze"]
    run_seed = _int_with_default(_run_meta(meta).get("run_seed"), 0)
    out: list[dict[str, Any]] = []
    for idx, (fn, name) in enumerate(zip(rule_fns, rule_names)):
        rng = np.random.default_rng((run_seed * 1_000_003 + 97 * (idx + 1)) % (2**63 - 1))
        try:
            opt_order = np.asarray(fn(pref_table, rng=rng), dtype=np.int64)
            if opt_order.size <= 0:
                continue
            winning_option_id = int(opt_order[0])
            color_ordering = np.asarray(options[winning_option_id], dtype=np.int64)
            out.append(
                {
                    "rule_idx": int(idx),
                    "rule_name": str(name),
                    "winning_option_id": int(winning_option_id),
                    "color_ordering": color_ordering,
                }
            )
        except (TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            warnings.warn(
                f"Skipping static power-direction baseline for rule '{name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
    return out

def _draw_power_direction_panel(*, ax, power_dirs: list[dict[str, Any]], current_rule_idx: int | None, num_colors: int) -> None:
    ax.set_title("Static Power Directions\n(All Self-Regarding)")
    if not power_dirs or num_colors <= 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Power baselines unavailable", ha="center", va="center")
        return
    n = len(power_dirs)
    ax.set_xlim(0.0, float(num_colors + 1.9))
    ax.set_ylim(-0.5, float(n - 0.5))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(np.arange(n))
    labels = []
    for item in power_dirs:
        nm = str(item.get("rule_name", "rule"))
        ridx = int(item.get("rule_idx", -1))
        labels.append(f"{nm}{' *' if current_rule_idx is not None and ridx == int(current_rule_idx) else ''}")
    ax.set_yticklabels(labels, fontsize=8)
    for row, item in enumerate(power_dirs):
        ordering = np.asarray(item.get("color_ordering", []), dtype=int)
        for rank in range(min(num_colors, ordering.size)):
            c_idx = int(ordering[rank])
            face = _sim_color(c_idx)
            rect = plt.Rectangle(
                (0.9 + rank, row - 0.35),
                0.9,
                0.7,
                facecolor=face,
                edgecolor="black" if int(item.get("rule_idx", -1)) == int(current_rule_idx or -99) else "#666666",
                linewidth=1.8 if int(item.get("rule_idx", -1)) == int(current_rule_idx or -99) else 0.8,
            )
            ax.add_patch(rect)
            r, g, b, _ = to_rgba(face)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            tcol = "black" if lum > 0.55 else "white"
            ax.text(0.9 + rank + 0.45, row, f"{c_idx}", ha="center", va="center", fontsize=7, color=tcol, fontweight="bold")
            ax.text(0.9 + rank + 0.45, row - 0.46, f"{rank+1}", ha="center", va="top", fontsize=6, color="#333333")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def _compute_area_puzzle_power_distances(
    *,
    area_series: pd.DataFrame,
    num_colors: int,
    power_ordering_current_rule: np.ndarray | None,
    meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    xlen = int(len(area_series))
    out = {
        "dist_outcome_power": np.full(xlen, np.nan, dtype=np.float32),
        "dist_puzzle_power": np.full(xlen, np.nan, dtype=np.float32),
        "dist_grid_power": np.full(xlen, np.nan, dtype=np.float32),
    }
    if power_ordering_current_rule is None or num_colors <= 0 or xlen <= 0:
        return out
    power_ord = np.asarray(power_ordering_current_rule, dtype=np.int64).reshape(-1)
    if power_ord.size != int(num_colors):
        return out
    dist_func = _summary_ordering_distance_func(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))
    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)

    color_cols = [f"area_color_{i}" for i in range(num_colors) if f"area_color_{i}" in area_series.columns]
    puzzle_cols = [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_series.columns]
    has_grid_ids = "grid_ordering_id" in area_series.columns
    has_puzzle_ids = "puzzle_ordering_id" in area_series.columns
    grid_prev = None
    puzzle_prev = None
    tie_rng = np.random.default_rng(_int_with_default(_run_meta(meta).get("run_seed"), 0) + 4242)

    win_ids = area_series["winning_option_id"].to_numpy(dtype=int) if "winning_option_id" in area_series.columns else np.full(xlen, -1, dtype=int)
    for i in range(xlen):
        oid = int(win_ids[i])
        if 0 <= oid < int(options.shape[0]):
            out["dist_outcome_power"][i] = np.float32(float(dist_func(np.asarray(options[oid], dtype=np.int64), power_ord, search_pairs)))
        pord = None
        if has_puzzle_ids:
            pid = int(area_series.iloc[i]["puzzle_ordering_id"])
            if 0 <= pid < int(options.shape[0]):
                pord = np.asarray(options[pid], dtype=np.int64)
        elif len(puzzle_cols) == num_colors:
            pvals = area_series.loc[area_series.index[i], puzzle_cols].to_numpy(dtype=float)
            if np.isfinite(pvals).all():
                pord = _summary_ordering_from_distribution_tie_aware(pvals, reference_ordering=puzzle_prev, rng=tie_rng)
        if pord is not None:
            out["dist_puzzle_power"][i] = np.float32(float(dist_func(pord, power_ord, search_pairs)))
            puzzle_prev = pord
        gord = None
        if has_grid_ids:
            gid = int(area_series.iloc[i]["grid_ordering_id"])
            if 0 <= gid < int(options.shape[0]):
                gord = np.asarray(options[gid], dtype=np.int64)
        elif len(color_cols) == num_colors:
            gvals = area_series.loc[area_series.index[i], color_cols].to_numpy(dtype=float)
            if np.isfinite(gvals).all():
                gord = _summary_ordering_from_distribution_tie_aware(gvals, reference_ordering=grid_prev, rng=tie_rng)
        if gord is not None:
            out["dist_grid_power"][i] = np.float32(float(dist_func(gord, power_ord, search_pairs)))
            grid_prev = gord
    return out

def _compute_area_puzzle_anti_monopoly_gate_metrics(
    *,
    area_series: pd.DataFrame,
    dist_decomp: dict[str, np.ndarray],
) -> dict[str, Any]:
    xlen = int(len(area_series))
    thresholds = {
        "conflict_min_dist": float(DEFAULT_SCORING_THRESHOLDS["puzzle_conflict_min_dist"]),
        "min_conflict_share": float(DEFAULT_SCORING_THRESHOLDS["min_puzzle_conflict_step_share_for_gate"]),
        "max_dominance_share_conflict": float(DEFAULT_SCORING_THRESHOLDS["max_puzzle_dominance_share_conflict"]),
        "min_recovery_share_conflict": float(DEFAULT_SCORING_THRESHOLDS["min_power_recovery_share_conflict"]),
    }
    out: dict[str, Any] = {
        **thresholds,
        "d_out_puz": np.full(xlen, np.nan, dtype=np.float32),
        "d_out_pow": np.full(xlen, np.nan, dtype=np.float32),
        "d_puz_pow": np.full(xlen, np.nan, dtype=np.float32),
        "margin": np.full(xlen, np.nan, dtype=np.float32),
        "conflict_mask": np.zeros(xlen, dtype=bool),
        "puzzle_max_share": np.full(xlen, np.nan, dtype=np.float32),
        "puzzle_entropy_norm": np.full(xlen, np.nan, dtype=np.float32),
        "puzzle_conflict_step_share": np.nan,
        "puzzle_dominance_share_conflict": np.nan,
        "power_recovery_share_conflict": np.nan,
        "puzzle_power_margin_mean_conflict": np.nan,
        "gate_puzzle_anti_monopoly": True,
        "puzzle_metric_available": False,
        "gate_conflict_eligible": False,
        "valid_rows": 0,
        "conflict_rows": 0,
    }
    if xlen <= 0:
        return out

    d_out_puz = area_series.get("puzzle_distance", pd.Series(np.nan, index=area_series.index)).to_numpy(dtype=float)
    d_out_pow = np.asarray(dist_decomp.get("dist_outcome_power", np.full(xlen, np.nan, dtype=np.float32)), dtype=float)
    d_puz_pow = np.asarray(dist_decomp.get("dist_puzzle_power", np.full(xlen, np.nan, dtype=np.float32)), dtype=float)
    if d_out_pow.size != xlen:
        d_out_pow = np.full(xlen, np.nan, dtype=float)
    if d_puz_pow.size != xlen:
        d_puz_pow = np.full(xlen, np.nan, dtype=float)

    margin = d_out_pow - d_out_puz
    valid = np.isfinite(d_out_puz) & np.isfinite(d_out_pow) & np.isfinite(d_puz_pow)
    conflict_mask = valid & (d_puz_pow >= float(thresholds["conflict_min_dist"]))

    out["d_out_puz"] = d_out_puz.astype(np.float32)
    out["d_out_pow"] = d_out_pow.astype(np.float32)
    out["d_puz_pow"] = d_puz_pow.astype(np.float32)
    out["margin"] = margin.astype(np.float32)
    out["conflict_mask"] = conflict_mask.astype(bool)
    out["valid_rows"] = int(np.count_nonzero(valid))
    out["conflict_rows"] = int(np.count_nonzero(conflict_mask))

    if int(np.count_nonzero(valid)) > 0:
        out["puzzle_conflict_step_share"] = float(np.mean(conflict_mask[valid]))
    if int(np.count_nonzero(conflict_mask)) > 0:
        conflict_margins = margin[conflict_mask]
        conflict_margins = conflict_margins[np.isfinite(conflict_margins)]
        if conflict_margins.size > 0:
            out["puzzle_dominance_share_conflict"] = float(np.mean(conflict_margins > 0.0))
            out["power_recovery_share_conflict"] = float(np.mean(conflict_margins < 0.0))
            out["puzzle_power_margin_mean_conflict"] = float(np.mean(conflict_margins))

    metrics_available = (
        np.isfinite(float(out["puzzle_conflict_step_share"]))
        and np.isfinite(float(out["puzzle_dominance_share_conflict"]))
        and np.isfinite(float(out["power_recovery_share_conflict"]))
    )
    enough_conflict = (
        metrics_available
        and float(out["puzzle_conflict_step_share"]) >= float(thresholds["min_conflict_share"])
    )
    anti_monopoly_ok = (
        metrics_available
        and float(out["puzzle_dominance_share_conflict"]) <= float(thresholds["max_dominance_share_conflict"])
        and float(out["power_recovery_share_conflict"]) >= float(thresholds["min_recovery_share_conflict"])
    )
    out["puzzle_metric_available"] = bool(metrics_available)
    out["gate_conflict_eligible"] = bool(enough_conflict)
    out["gate_puzzle_anti_monopoly"] = bool(anti_monopoly_ok if enough_conflict else True)

    puzzle_cols = sorted([c for c in area_series.columns if c.startswith("puzzle_color_")], key=lambda c: int(c.split("_")[-1]))
    if puzzle_cols:
        pdata = area_series[puzzle_cols].to_numpy(dtype=float)
        n_cols = int(len(puzzle_cols))
        max_share = np.full(xlen, np.nan, dtype=np.float32)
        entropy_norm = np.full(xlen, np.nan, dtype=np.float32)
        for i in range(xlen):
            row = pdata[i]
            if not np.isfinite(row).all():
                continue
            s = float(np.sum(row))
            if s <= 0.0:
                continue
            p = row / s
            max_share[i] = np.float32(float(np.max(p)))
            if n_cols > 1:
                h = float(-np.sum(p * np.log(p + 1e-15)))
                entropy_norm[i] = np.float32(h / float(np.log(float(n_cols))))
        out["puzzle_max_share"] = max_share
        out["puzzle_entropy_norm"] = entropy_norm
    return out

def _render_area_puzzle_gate_page(
    *,
    pdf: PdfPages,
    area_series: pd.DataFrame,
    dist_decomp: dict[str, np.ndarray],
    suptitle: str,
    include_overview: bool = True,
    include_decomposition: bool = True,
) -> None:
    metrics = _compute_area_puzzle_anti_monopoly_gate_metrics(
        area_series=area_series,
        dist_decomp=dist_decomp,
    )
    x = area_series["step"].to_numpy(dtype=float)

    status_ok = bool(metrics["gate_puzzle_anti_monopoly"])
    status_txt = "PASS" if status_ok else "FAIL"
    status_color = "tab:green" if status_ok else "tab:red"
    conf_eligible = bool(metrics["gate_conflict_eligible"])
    metric_ready = bool(metrics["puzzle_metric_available"])

    d_out_puz = np.asarray(metrics["d_out_puz"], dtype=float)
    d_out_pow = np.asarray(metrics["d_out_pow"], dtype=float)
    d_puz_pow = np.asarray(metrics["d_puz_pow"], dtype=float)
    d_grid_pow = np.asarray(dist_decomp.get("dist_grid_power", np.full_like(d_puz_pow, np.nan, dtype=float)), dtype=float)
    conflict_mask = np.asarray(metrics["conflict_mask"], dtype=bool)
    margin = np.asarray(metrics["margin"], dtype=float)

    # Page A: gate status + puzzle concentration signals.
    if include_overview:
        figa, axa = plt.subplots(1, 2, figsize=(11.69, 8.27))
        axa = np.asarray(axa).ravel()

        axa[0].axis("off")
        axa[0].text(
            0.02,
            0.98,
            "Puzzle Anti-Monopoly Gate",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        axa[0].text(
            0.02,
            0.80,
            status_txt,
            ha="left",
            va="top",
            fontsize=24,
            color=status_color,
            fontweight="bold",
        )
        lines = [
            f"metrics_available: {metric_ready}",
            f"conflict_eligible: {conf_eligible}",
            f"conflict_share: {float(metrics['puzzle_conflict_step_share']):.3f} (>= {float(metrics['min_conflict_share']):.3f})",
            f"dominance_share_conflict: {float(metrics['puzzle_dominance_share_conflict']):.3f} (<= {float(metrics['max_dominance_share_conflict']):.3f})",
            f"recovery_share_conflict: {float(metrics['power_recovery_share_conflict']):.3f} (>= {float(metrics['min_recovery_share_conflict']):.3f})",
            f"margin_mean_conflict: {float(metrics['puzzle_power_margin_mean_conflict']):.3f}",
            f"valid_rows: {int(metrics['valid_rows'])}",
            f"conflict_rows: {int(metrics['conflict_rows'])}",
        ]
        axa[0].text(0.02, 0.60, "\n".join(lines), ha="left", va="top", fontsize=9)

        max_share = np.asarray(metrics["puzzle_max_share"], dtype=float)
        entropy = np.asarray(metrics["puzzle_entropy_norm"], dtype=float)
        if np.isfinite(max_share).any():
            axa[1].plot(x, max_share, color="tab:red", linewidth=1.4, label="max puzzle color share")
        if np.isfinite(entropy).any():
            ax1b = axa[1].twinx()
            ax1b.plot(x, entropy, color="tab:blue", linestyle="--", linewidth=1.2, label="puzzle entropy (norm)")
            ax1b.set_ylim(-0.02, 1.02)
            ax1b.set_ylabel("entropy [0..1]", color="tab:blue")
            ax1b.tick_params(axis="y", colors="tab:blue")
            h1, l1 = axa[1].get_legend_handles_labels()
            h2, l2 = ax1b.get_legend_handles_labels()
            if h1 or h2:
                axa[1].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        elif len(axa[1].lines) > 0:
            axa[1].legend(loc="best", fontsize=8)
        axa[1].set_title("Puzzle Concentration Signals")
        axa[1].set_ylabel("max share [0..1]", color="tab:red")
        axa[1].tick_params(axis="y", colors="tab:red")
        _set_unit_ylim_visible(axa[1])
        axa[1].grid(True, alpha=0.25)
        axa[1].set_xlabel("step")

        figa.suptitle(suptitle + " | Puzzle Anti-Monopoly Gate (Overview)", fontsize=11)
        figa.tight_layout()
        pdf.savefig(figa, dpi=140)
        plt.close(figa)

    # Page B: split decomposition (3 panels) to avoid overload.
    if include_decomposition:
        figb, axb = plt.subplots(3, 1, figsize=(11.69, 8.27), sharex=True)
        axb = np.asarray(axb).ravel()

        if np.isfinite(d_out_puz).any() or np.isfinite(d_out_pow).any():
            if np.isfinite(d_out_puz).any():
                axb[0].plot(x, d_out_puz, color="black", linestyle="--", linewidth=1.3, label="outcome↔puzzle")
            if np.isfinite(d_out_pow).any():
                axb[0].plot(x, d_out_pow, color="tab:red", linewidth=1.2, label="outcome↔power")
            if conflict_mask.any():
                axb[0].fill_between(x, 0.0, 1.0, where=conflict_mask, color="tab:blue", alpha=0.06, step="mid")
            if len(axb[0].lines) > 0:
                axb[0].legend(loc="best", fontsize=8)
        else:
            axb[0].text(0.5, 0.5, "Outcome distance series unavailable", ha="center", va="center")
            axb[0].set_yticks([])
        axb[0].set_title("Puzzle / Power Distance Decomposition A: Outcome↔Puzzle and Outcome↔Power")
        axb[0].set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(axb[0])

        has_pairwise = False
        if np.isfinite(d_puz_pow).any():
            has_pairwise = True
            axb[1].plot(x, d_puz_pow, color="tab:blue", linewidth=1.2, label="puzzle↔power")
        if np.isfinite(d_grid_pow).any():
            has_pairwise = True
            axb[1].plot(x, d_grid_pow, color="tab:orange", linewidth=1.2, linestyle=":", label="grid↔power")
        axb[1].axhline(
            float(metrics["conflict_min_dist"]),
            color="tab:blue",
            linestyle=":",
            linewidth=1.15,
            label="conflict threshold",
        )
        if conflict_mask.any():
            axb[1].fill_between(x, 0.0, 1.0, where=conflict_mask, color="tab:blue", alpha=0.06, step="mid")
        if has_pairwise:
            axb[1].legend(loc="best", fontsize=8)
        else:
            axb[1].text(0.5, 0.5, "Pairwise puzzle/power diagnostics unavailable", ha="center", va="center")
            axb[1].set_yticks([])
        axb[1].set_title("Puzzle / Power Distance Decomposition B: Puzzle↔Power and Grid↔Power")
        axb[1].set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(axb[1])

        if np.isfinite(margin).any():
            axb[2].plot(x, margin, color="purple", linewidth=1.15, label="margin = outcome↔power - outcome↔puzzle")
            axb[2].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
            if conflict_mask.any():
                axb[2].scatter(
                    x[conflict_mask],
                    margin[conflict_mask],
                    s=9,
                    color="purple",
                    alpha=0.65,
                    label="conflict steps",
                )
            if len(axb[2].lines) > 0:
                axb[2].legend(loc="best", fontsize=8)
        else:
            axb[2].text(0.5, 0.5, "Margin unavailable", ha="center", va="center")
            axb[2].set_yticks([])
        axb[2].set_title("Puzzle / Power Distance Decomposition C: Margin Around Puzzle vs Power")
        axb[2].set_ylabel("margin [-1..1]")
        axb[2].set_ylim(-1.02, 1.02)

        for a in axb:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")

        figb.suptitle(suptitle + " | Puzzle Anti-Monopoly Gate (Distance Decomposition)", fontsize=11)
        figb.tight_layout()
        pdf.savefig(figb, dpi=140)
        plt.close(figb)

def _compute_group_puzzle_opportunity_distances(
    *,
    area_series: pd.DataFrame,
    personality_groups: np.ndarray,
    num_colors: int,
    meta: dict[str, Any],
) -> pd.DataFrame:
    """Per-step distance between puzzle ordering and each personality-group ordering."""
    if num_colors <= 0 or personality_groups.ndim != 2:
        return pd.DataFrame()
    puzzle_cols = [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_series.columns]
    has_puzzle_ids = "puzzle_ordering_id" in area_series.columns
    if (not has_puzzle_ids and len(puzzle_cols) != num_colors) or area_series.empty:
        return pd.DataFrame()
    dist_func = _summary_ordering_distance_func(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))
    x = area_series["step"].to_numpy(dtype=int)
    out = pd.DataFrame({"step": x.astype(np.int32)})
    prev_ord = None
    tie_rng = np.random.default_rng(_int_with_default(_run_meta(meta).get("run_seed"), 0) + 7171)
    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=int)
    puzzle_orders: list[np.ndarray | None] = []
    for _, r in area_series.iterrows():
        pord = None
        if has_puzzle_ids:
            pid = int(r.get("puzzle_ordering_id", -1))
            if 0 <= pid < int(options.shape[0]):
                pord = np.asarray(options[pid], dtype=np.int64)
        elif len(puzzle_cols) == num_colors:
            vals = r[puzzle_cols].to_numpy(dtype=float)
            if np.isfinite(vals).all():
                pord = _summary_ordering_from_distribution_tie_aware(vals, reference_ordering=prev_ord, rng=tie_rng)
        if pord is not None:
            prev_ord = pord
            puzzle_orders.append(pord)
        else:
            puzzle_orders.append(None)
    n_groups = int(personality_groups.shape[0])
    for gi in range(n_groups):
        target = np.asarray(personality_groups[gi][:num_colors], dtype=np.int64)
        vals = np.full(len(puzzle_orders), np.nan, dtype=np.float32)
        for i, pord in enumerate(puzzle_orders):
            if pord is None:
                continue
            vals[i] = np.float32(float(dist_func(np.asarray(pord, dtype=np.int64), target, search_pairs)))
        out[f"group_{gi}_puzzle_opp_dist"] = vals
    return out

def _build_elected_ordering_background_image(
    *,
    winning_option_ids: np.ndarray,
    num_colors: int,
    alpha: float = 0.28,
) -> np.ndarray | None:
    """Build RGBA image for elected ordering background in quality-distance plots."""
    ids = np.asarray(winning_option_ids, dtype=int).reshape(-1)
    n_steps = int(ids.size)
    if n_steps <= 0 or num_colors <= 0:
        return None

    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=int)
    rgba = np.zeros((int(num_colors), n_steps, 4), dtype=np.float32)

    for t, oid in enumerate(ids.tolist()):
        if oid < 0 or oid >= int(options.shape[0]):
            # transparent for missing/invalid winner rows
            rgba[:, t, :] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            continue
        ordering = options[int(oid)]
        for rank in range(int(num_colors)):
            c_idx = int(ordering[rank])
            r, g, b, _ = to_rgba(_sim_color(c_idx))
            rgba[rank, t, :] = np.array([r, g, b, float(alpha)], dtype=np.float32)
    return rgba

def _draw_personality_group_order_block(*, ax, personality_groups: np.ndarray, num_colors: int) -> None:
    ax.set_title("Personality Group -> Color Preference Order")
    if personality_groups.ndim != 2 or personality_groups.shape[0] == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No personality group metadata", ha="center", va="center")
        return

    n_groups = int(personality_groups.shape[0])
    n_slots = int(min(num_colors, personality_groups.shape[1]))
    ax.set_xlim(0.0, float(n_slots + 1.8))
    ax.set_ylim(-0.5, float(n_groups - 0.5))
    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels([f"g{i}" for i in range(n_groups)])
    for gi, tick in enumerate(ax.get_yticklabels()):
        c = get_group_color(int(gi))
        r, g, b, _ = to_rgba(c)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "black" if luminance > 0.55 else "white"
        tick.set_color(txt)
        tick.set_bbox(
            dict(
                facecolor=c,
                edgecolor="none",
                boxstyle="round,pad=0.20,rounding_size=0.08",
                alpha=0.95,
            )
        )
    ax.set_xticks([])
    ax.grid(False)

    for gi in range(n_groups):
        order = personality_groups[gi].astype(int).tolist()
        for pos, color_id in enumerate(order[:n_slots]):
            x0 = float(pos + 1.0)
            y0 = float(gi - 0.32)
            rect = plt.Rectangle(
                (x0, y0),
                0.9,
                0.64,
                facecolor=_sim_color(color_id),
                edgecolor="black",
                linewidth=0.6,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + 0.45,
                y0 - 0.06,
                str(pos + 1),
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
            )
            ax.text(
                x0 + 0.45,
                y0 + 0.32,
                str(int(color_id)),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                fontweight="bold",
            )

    for s in ax.spines.values():
        s.set_alpha(0.3)
