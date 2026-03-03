from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import numpy as np
import hashlib
import copy


RULE_LABELS = {
    0: "majority",
    1: "approval",
    2: "utilitarian",
    3: "borda",
    4: "random",
}

INTEGER_DOE_KEYS: set[str] = {"known_cells"}


def _known_cells_probe_bounds(model_cfg: dict[str, Any]) -> tuple[float, float]:
    num_colors = int(model_cfg["num_colors"])
    grid_fields = int(model_cfg["height"]) * int(model_cfg["width"])
    max_known = max(num_colors, int(np.floor(0.02 * float(grid_fields))))
    return float(num_colors), float(max_known)


# Phase-1 DOE ranges (confirmed)
DEFAULT_DOE_RANGES: dict[str, tuple[float, float]] = {
    "election_cost_rate": (0.001, 0.10),
    "reward_rate_personal": (0.00, 0.30),
    "break_even_distance_common": (0.15, 0.45),
    "election_impact_on_mutation": (1.0, 3.0),
    "mu": (0.15, 1.00),
    "participation_alpha": (0.01, 0.20),
    "participation_beta": (2.5, 9.5),
    "participation_init_q": (0.15, 1.2),
    "altruism_static": (0.25, 0.75),
}


# Frozen model settings for DOE phase-1
DEFAULT_FROZEN_MODEL: dict[str, Any] = {
    "distance_idx": 0,
    "participation_q_max": 2.0,
    "bias_toward_participation": 0.0,
    "altruism_mode": "satisfaction",
    "altruism_response_gamma": 1.0,
    "altruism_learning": False,
    "altruism_alpha": 0.05,
    "altruism_init": 0.5,
    "altruism_clip_min": 0.0,
    "altruism_clip_max": 1.0,
    "satisfaction_mode": "area",
    "satisfaction_baseline_alpha": 0.1,
    "quality_target_mode": "puzzle",
    "puzzle_local_kappa": 30.0,
    "puzzle_shock_prob": 0.05,
    "participation_baseline_alpha": 0.1,
    "initial_agent_assets": 100.0,
    "heterogeneity": 0.3,
    "known_cells": 10,
    "personal_preference_peakedness": 1.0,
    "num_agents": 100,
    "num_colors": 4,
    "num_personality_groups": 4,
    "height": 30,
    "width": 50,
    "num_areas": 1,
    "av_area_height": 30,
    "av_area_width": 50,
    "area_size_variance": 0.0,
    "color_patches_steps": 0,
    "patch_power": 1.0,
}


DEFAULT_FROZEN_SIM: dict[str, Any] = {
    "runs": 1,
    "num_steps": 250,
    "store_grid": False,
    "grid_interval": 1,
}


DEFAULT_DOE_PROFILES: dict[str, dict[str, Any]] = {
    "phase1": {
        "name": "phase1",
        "ranges": dict(DEFAULT_DOE_RANGES),
        "frozen_model": dict(DEFAULT_FROZEN_MODEL),
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase2_altruism_learning": {
        "name": "phase2_altruism_learning",
        "ranges": {
            "election_cost_rate": (0.001, 0.10),
            "reward_rate_personal": (0.00, 0.30),
            "break_even_distance_common": (0.15, 0.45),
            "election_impact_on_mutation": (1.0, 3.0),
            "mu": (0.15, 1.00),
            "participation_alpha": (0.01, 0.20),
            "participation_beta": (2.5, 9.5),
            "participation_init_q": (0.15, 1.2),
            "altruism_alpha": (0.01, 0.08),
            "altruism_init": (0.20, 0.80),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "surprise_learning",
            "altruism_learning": True,
        },
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase2_altruism_probe": {
        "name": "phase2_altruism_probe",
        "ranges": {
            "election_cost_rate": (0.001, 0.10),
            "reward_rate_personal": (0.00, 0.30),
            "break_even_distance_common": (0.15, 0.45),
            "election_impact_on_mutation": (1.0, 3.0),
            "mu": (0.15, 1.00),
            "participation_alpha": (0.01, 0.20),
            "participation_beta": (2.5, 9.5),
            "participation_init_q": (0.15, 1.2),
            "altruism_alpha": (0.01, 0.08),
            "altruism_init": (0.20, 0.80),
            "satisfaction_baseline_alpha": (0.02, 0.25),
            "known_cells": _known_cells_probe_bounds(DEFAULT_FROZEN_MODEL),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "surprise_learning",
            "altruism_learning": True,
        },
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase3_puzzle_main": {
        "name": "phase3_puzzle_main",
        "ranges": {
            "election_cost_rate": (0.001, 0.05),
            "reward_rate_personal": (0.03, 0.25),
            "break_even_distance_common": (0.25, 0.60),
            "election_impact_on_mutation": (0.75, 3.0),
            "mu": (0.05, 0.50),
            "participation_alpha": (0.03, 0.20),
            "participation_beta": (2.5, 10.0),
            "participation_init_q": (0.01, 1.20),
            "known_cells": _known_cells_probe_bounds(DEFAULT_FROZEN_MODEL),
            "altruism_response_gamma": (0.9, 1.00),
            "puzzle_local_kappa": (5.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.15),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase3_puzzle_refine1": {
        "name": "phase3_puzzle_refine1",
        # Refined from DOE: data/simulation_output/doe_20260222_055633 (250 steps, puzzle mode)
        # Goal: reduce turnout saturation while keeping puzzle/process knobs exploratory.
        "ranges": {
            "election_cost_rate": (0.020, 0.045),
            "reward_rate_personal": (0.050, 0.160),
            "break_even_distance_common": (0.33, 0.52),
            "election_impact_on_mutation": (0.90, 2.20),
            "mu": (0.05, 0.50),
            "participation_alpha": (0.07, 0.18),
            "participation_beta": (3.5, 9.0),
            # Strongest signal in 250-step DOE: high init_q drove turnout saturation.
            "participation_init_q": (0.05, 0.55),
            "known_cells": (5.0, 27.0),
            # We keep near-direct satisfaction response; effect looked weak in current range.
            "altruism_response_gamma": (0.75, 1.00),
            # We keep puzzle dynamics reasonably broad in refine1 (not enough evidence yet to clamp hard).
            "puzzle_local_kappa": (15.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.135),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_puzzle_summary_dev": {
        "name": "phase3_puzzle_summary_dev",
        # Smaller DOE for summary/visual iteration after puzzle refactor.
        # Bias toward viable regimes, but retain enough spread for varied plot examples.
        "ranges": {
            "election_cost_rate": (0.025, 0.045),
            "reward_rate_personal": (0.055, 0.145),
            "break_even_distance_common": (0.35, 0.50),  # !!! 
            "election_impact_on_mutation": (0.95, 2.00),
            "mu": (0.10, 0.45),
            "participation_alpha": (0.09, 0.17),
            "participation_beta": (4.0, 8.5),
            "participation_init_q": (0.08, 0.45),
            "known_cells": (10.0, 24.0),
            "puzzle_local_kappa": (20.0, 75.0),
            "puzzle_shock_prob": (0.01, 0.10),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "altruism_static": 1,
            "altruism_response_gamma": 1.0,
            "altruism_satisfaction_theta": 0.7,
            "altruism_satisfaction_slope": 10.0,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.5,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 100,
        },
    },
    "phase3_puzzle_motion_hypothesis": {
        "name": "phase3_puzzle_motion_hypothesis",
        # Focused hypothesis DOE:
        # Test whether jumpier / less persistent puzzle motion reduces puzzle dominance.
        # Vary puzzle motion + the strongest observed confounds, keep the rest near a viable center.
        "ranges": {
            "known_cells": (4.0, 27.0),
            "election_impact_on_mutation": (0.90, 2.80),
            "puzzle_local_kappa": (1.0, 120.0),
            "puzzle_shock_prob": (0.00, 0.40),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "altruism_response_gamma": 1.0,
            # Freeze the post-Stage-A/majority-confirmed mapping while testing puzzle-motion effects.
            "altruism_satisfaction_theta": 0.7,
            "altruism_satisfaction_slope": 2.0,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
            # Refine1-like center values to isolate the puzzle-motion hypothesis.
            "election_cost_rate": 0.035,
            "reward_rate_personal": 0.12,
            "break_even_distance_common": 0.425,
            "mu": 0.30,
            "participation_alpha": 0.14,
            "participation_beta": 6.5,
            "participation_init_q": 0.30,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 250,
        },
    },
    "phase3_puzzle_refine2": {
    "name": "phase3_puzzle_refine2",
    # Refine1 + targeted re-widening after broadcheck audit on current participation signal mode.
    "ranges": {
        # re-widened (boundary pressure in broadcheck)
        "election_cost_rate": (0.015, 0.055),
        "reward_rate_personal": (0.050, 0.240),
        "break_even_distance_common": (0.30, 0.58),
        "election_impact_on_mutation": (0.80, 2.80),
        "participation_init_q": (0.03, 0.70),
        "puzzle_shock_prob": (0.00, 0.15),

        # unchanged from refine1
        "mu": (0.05, 0.50),
        "participation_alpha": (0.07, 0.18),
        "participation_beta": (3.5, 9.0),
        "known_cells": (5.0, 27.0),
        "altruism_response_gamma": (0.75, 1.00),
        "puzzle_local_kappa": (15.0, 80.0),
    },
    "frozen_model": {
        **DEFAULT_FROZEN_MODEL,
        "altruism_mode": "satisfaction",
        "altruism_satisfaction_theta": 0.7,
        "altruism_satisfaction_slope": 2.0,
        "altruism_learning": False,
        "participation_signal_mode": "group_centered_delta_rel_plus_fee",
        "participation_signal_fee_weight": 1.0,
        "participation_signal_group_shrink_k": 10.0,
        "participation_signal_clip": 0.25,
    },
    "frozen_simulation": {
        **DEFAULT_FROZEN_SIM,
        "num_steps": 300,
    },
    },
    "phase3_puzzle_refine3": {
        "name": "phase3_puzzle_refine3",
        "ranges": {
            # Turnout-start / turnout-shape focused
            "participation_init_q": (0.10, 0.45),
            "participation_beta": (3.5, 7.5),
            "participation_alpha": (0.08, 0.18),
            "election_cost_rate": (0.015, 0.050),

            # Reward / punishment gate balance
            "reward_rate_personal": (0.060, 0.220),
            "break_even_distance_common": (0.32, 0.58),

            # Puzzle dominance levers
            "known_cells": (4.0, 15.0),
            "election_impact_on_mutation": (1.00, 2.80),

            # Puzzle motion (secondary; keep conservative shock upper bound)
            "puzzle_local_kappa": (15.0, 70.0),
            "puzzle_shock_prob": (0.00, 0.06),

            # General dynamics
            "mu": (0.08, 0.45),
            "altruism_response_gamma": (0.80, 1.00),

            # Re-open narrow local window for mapping confirmation
            "altruism_satisfaction_theta": (0.65, 0.75),
            "altruism_satisfaction_slope": (1.5, 4.0),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_turnout_focus_medium": {
        "name": "phase3_turnout_focus_medium",
        "ranges": {
            # Constrain startup turnout drivers to target a realistic initial window.
            "participation_init_q": (0.11, 0.18),
            "participation_beta": (3.8, 6.2),

            # Primary turnout-drop / participation dynamics levers.
            "election_cost_rate": (0.010, 0.035),
            "participation_alpha": (0.07, 0.14),
            "reward_rate_personal": (0.10, 0.24),
            "break_even_distance_common": (0.34, 0.62),

            # Keep puzzle/power balance in a reasonable regime.
            "known_cells": (4.0, 12.0),
            "election_impact_on_mutation": (1.20, 2.80),
            "puzzle_local_kappa": (25.0, 75.0),
            "puzzle_shock_prob": (0.00, 0.03),

            # General dynamics.
            "mu": (0.08, 0.40),
            "altruism_response_gamma": (0.90, 1.00),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_satisfaction_theta": 0.70,
            "altruism_satisfaction_slope": 2.0,
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_refine4": {
        "name": "phase3_refine4",
        # Refine3 + turnout-focused constraints from phase3_turnout_focus_medium partial run
        # + alpha=2 puzzle-generator regime assumptions.
        "ranges": {
            # Startup turnout constraints (successful in turnout_focus_medium)
            "participation_init_q": (0.11, 0.18),
            "participation_beta": (3.8, 6.0),

            # Turnout-drop / participation dynamics levers (focused)
            "election_cost_rate": (0.010, 0.030),
            "participation_alpha": (0.07, 0.12),
            "reward_rate_personal": (0.12, 0.26),
            "break_even_distance_common": (0.34, 0.62),

            # Puzzle / power balance in alpha=2 generator regime
            "known_cells": (4.0, 12.0),
            "election_impact_on_mutation": (1.20, 2.60),
            "puzzle_local_kappa": (30.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.03),

            # General dynamics
            "mu": (0.08, 0.35),
            "altruism_response_gamma": (0.90, 1.00),

            # Keep mapping in the empirically good neighborhood (slight local re-open)
            "altruism_satisfaction_theta": (0.68, 0.74),
            "altruism_satisfaction_slope": (1.6, 2.8),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_refine5_local": {
        "name": "phase3_refine5_local",
        # Local search around refine4 with explicit focus on restoring
        # inter-group participation-composition dynamics.
        "ranges": {
            # Startup turnout constraints (keep realistic starts).
            "participation_init_q": (0.11, 0.18),
            "participation_beta": (3.8, 6.0),

            # Participation dynamics levers.
            "election_cost_rate": (0.010, 0.030),
            "participation_alpha": (0.07, 0.14),
            "reward_rate_personal": (0.12, 0.26),
            "break_even_distance_common": (0.34, 0.62),

            # Puzzle / power balance.
            "known_cells": (4.0, 12.0),
            "election_impact_on_mutation": (1.20, 2.60),
            "puzzle_local_kappa": (30.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.03),

            # General dynamics.
            "mu": (0.08, 0.35),
            "altruism_response_gamma": (0.90, 1.00),

            # Satisfaction->altruism mapping neighborhood.
            "altruism_satisfaction_theta": (0.68, 0.74),
            "altruism_satisfaction_slope": (1.6, 2.8),

            # New local re-open: reduce fee dominance and allow stronger
            # group-differentiated participation updates.
            "participation_signal_fee_weight": (0.35, 1.00),
            "participation_signal_group_shrink_k": (1.0, 10.0),
            "participation_signal_clip": (0.20, 0.45),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_turnover_balance_probe_medium": {
        "name": "phase3_turnover_balance_probe_medium",
        # Focused medium probe for turnout-vs-turnover balance in the current regime.
        # Built from robust envelopes observed in doe_20260227_030932.
        "ranges": {
            "participation_signal_fee_weight": (0.40, 0.76),
            "participation_signal_group_shrink_k": (3.0, 8.9),
            "participation_signal_clip": (0.22, 0.31),
            "participation_alpha": (0.10, 0.13),
            "participation_beta": (5.3, 5.9),
            "participation_init_q": (0.14, 0.17),
            "election_cost_rate": (0.013, 0.024),
            "reward_rate_personal": (0.206, 0.245),
            "known_cells": (5.0, 9.0),
            "election_impact_on_mutation": (1.40, 2.05),
            "break_even_distance_common": (0.34, 0.60),
            "mu": (0.10, 0.35),
            "puzzle_local_kappa": (30.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.03),
            "altruism_satisfaction_theta": (0.68, 0.74),
            "altruism_satisfaction_slope": (1.6, 2.8),
            "altruism_response_gamma": (0.90, 1.00),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_turnover_balance_probe_medium_v2": {
        "name": "phase3_turnover_balance_probe_medium_v2",
        # Re-opened medium probe:
        # keep realistic startup turnout, but re-open key levers that can restore
        # stronger inter-group turnover dynamics beyond what refine5 already explored.
        "ranges": {
            # Participation-signal structure (main re-open).
            "participation_signal_fee_weight": (0.15, 0.95),
            "participation_signal_group_shrink_k": (0.0, 10.0),
            "participation_signal_clip": (0.18, 0.45),

            # Participation learning response shape.
            "participation_alpha": (0.08, 0.18),
            "participation_beta": (4.0, 7.0),
            "participation_init_q": (0.12, 0.18),

            # Incentive and gate balance.
            "election_cost_rate": (0.010, 0.035),
            "reward_rate_personal": (0.15, 0.28),
            "break_even_distance_common": (0.32, 0.65),

            # Puzzle/power interaction.
            "known_cells": (4.0, 12.0),
            "election_impact_on_mutation": (1.20, 2.80),
            "puzzle_local_kappa": (20.0, 90.0),
            "puzzle_shock_prob": (0.00, 0.05),
            "mu": (0.08, 0.40),

            # Altruism mapping local re-open.
            "altruism_satisfaction_theta": (0.66, 0.76),
            "altruism_satisfaction_slope": (1.4, 3.4),
            "altruism_response_gamma": (0.85, 1.00),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_refine5_party_relative_probe": {
        "name": "phase3_refine5_party_relative_probe",
        # Probe around refine5 envelope with party-relative participation signal mode.
        # Purpose: test whether stronger group-level competition emerges without
        # destabilizing viability/quality metrics.
        "ranges": {
            # Startup turnout constraints (keep realistic starts).
            "participation_init_q": (0.11, 0.18),
            "participation_beta": (3.8, 6.0),

            # Participation dynamics levers.
            "election_cost_rate": (0.010, 0.030),
            "participation_alpha": (0.07, 0.14),
            "reward_rate_personal": (0.12, 0.26),
            "break_even_distance_common": (0.34, 0.62),

            # Puzzle / power balance.
            "known_cells": (4.0, 12.0),
            "election_impact_on_mutation": (1.20, 2.60),
            "puzzle_local_kappa": (30.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.03),

            # General dynamics.
            "mu": (0.08, 0.35),
            "altruism_response_gamma": (0.90, 1.00),

            # Satisfaction->altruism mapping neighborhood.
            "altruism_satisfaction_theta": (0.68, 0.74),
            "altruism_satisfaction_slope": (1.6, 2.8),

            # Group-relative signal shaping.
            "participation_signal_group_shrink_k": (1.0, 10.0),
            "participation_signal_clip": (0.20, 0.45),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
            # Participant fee salience in party mode; keep fixed for comparability.
            "participation_signal_fee_weight": 1.0,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_refine6_party_relative_tuned": {
        "name": "phase3_refine6_party_relative_tuned",
        # Tuned follow-up to phase3_refine5_party_relative_probe
        # (data/simulation_output/doe_party_probe_20260227_194026).
        # Built around high-score/high-robustness neighborhoods, but with
        # deliberate re-widening on incentive knobs to test pressure regimes.
        "ranges": {
            # Startup turnout / response shape.
            "participation_init_q": (0.11, 0.18),
            "participation_alpha": (0.09, 0.14),
            "participation_beta": (4.1, 6.0),

            # Incentive knobs (intentionally widened vs party-probe robust core).
            "election_cost_rate": (0.008, 0.035),
            "reward_rate_personal": (0.13, 0.285),
            "break_even_distance_common": (0.32, 0.64),

            # Puzzle / power balance.
            "known_cells": (4.0, 12.0),
            "election_impact_on_mutation": (1.20, 2.60),
            "puzzle_local_kappa": (30.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.03),
            "mu": (0.08, 0.35),

            # Satisfaction -> altruism mapping.
            "altruism_satisfaction_theta": (0.68, 0.74),
            "altruism_satisfaction_slope": (1.6, 2.8),
            "altruism_response_gamma": (0.90, 1.00),

            # Party-relative signal shaping (bounded to robustness-friendly area).
            "participation_signal_group_shrink_k": (1.2, 4.0),
            "participation_signal_clip": (0.20, 0.40),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
            # Participant fee salience in party mode; keep fixed for comparability.
            "participation_signal_fee_weight": 1.0,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_party_broad_global_retune": {
        "name": "phase3_party_broad_global_retune",
        # Broad/global retune profile for the party learning regime (fee variant).
        # Purpose: one large DOE for validation + global optimization before final local refine/freeze.
        "ranges": {
            # Participation dynamics (re-opened globally, still centered around viable region).
            "participation_init_q": (0.10, 0.20),
            "participation_alpha": (0.07, 0.16),
            "participation_beta": (3.5, 6.5),
            "election_cost_rate": (0.006, 0.040),
            "reward_rate_personal": (0.10, 0.30),
            "break_even_distance_common": (0.28, 0.70),

            # Puzzle / power balance (explicitly widened for seed-regime robustness).
            "known_cells": (3.0, 16.0),
            "election_impact_on_mutation": (0.90, 3.20),
            "puzzle_local_kappa": (10.0, 110.0),
            "puzzle_shock_prob": (0.00, 0.08),
            "mu": (0.06, 0.45),

            # Satisfaction -> altruism mapping (moderately re-opened).
            "altruism_satisfaction_theta": (0.64, 0.78),
            "altruism_satisfaction_slope": (1.2, 3.6),
            "altruism_response_gamma": (0.85, 1.00),

            # Party-signal shaping.
            "participation_signal_fee_weight": (0.35, 1.0),
            "participation_signal_group_shrink_k": (0.5, 8.0),
            "participation_signal_clip": (0.18, 0.50),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_party_local_search_balanced_v1": {
        "name": "phase3_party_local_search_balanced_v1",
        # Local exploit around high-score + robust neighborhood from
        # doe_party_broad_global_retune_20260228_022651.
        "ranges": {
            "participation_init_q": (0.11, 0.19),
            "participation_alpha": (0.082, 0.155),
            "participation_beta": (3.80, 6.25),
            "election_cost_rate": (0.007, 0.034),
            "reward_rate_personal": (0.16, 0.29),
            "break_even_distance_common": (0.31, 0.66),
            "known_cells": (8.0, 16.0),
            "election_impact_on_mutation": (1.05, 3.05),
            "puzzle_local_kappa": (22.0, 105.0),
            "puzzle_shock_prob": (0.004, 0.075),
            "mu": (0.10, 0.42),
            "altruism_satisfaction_theta": (0.65, 0.765),
            "altruism_satisfaction_slope": (1.35, 3.50),
            "altruism_response_gamma": (0.865, 0.985),
            "participation_signal_fee_weight": (0.40, 0.90),
            "participation_signal_group_shrink_k": (1.0, 6.5),
            "participation_signal_clip": (0.21, 0.47),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_party_local_search_robust_v1": {
        "name": "phase3_party_local_search_robust_v1",
        # Robustness-tilted local exploit:
        # lower fee/cost pressure and avoid extreme known_cells upper tail.
        "ranges": {
            "participation_init_q": (0.11, 0.185),
            "participation_alpha": (0.09, 0.16),
            "participation_beta": (3.8, 6.2),
            "election_cost_rate": (0.007, 0.024),
            "reward_rate_personal": (0.20, 0.30),
            "break_even_distance_common": (0.34, 0.64),
            "known_cells": (7.0, 13.0),
            "election_impact_on_mutation": (1.10, 2.90),
            "puzzle_local_kappa": (24.0, 95.0),
            "puzzle_shock_prob": (0.005, 0.065),
            "mu": (0.10, 0.42),
            "altruism_satisfaction_theta": (0.65, 0.765),
            "altruism_satisfaction_slope": (1.35, 3.50),
            "altruism_response_gamma": (0.87, 0.99),
            "participation_signal_fee_weight": (0.35, 0.70),
            "participation_signal_group_shrink_k": (1.0, 4.5),
            "participation_signal_clip": (0.20, 0.45),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 300,
        },
    },
    "phase3_party_final_supersearch_v1": {
        "name": "phase3_party_final_supersearch_v1",
        # Final global search profile before thesis freeze.
        # Built from:
        # - broad + local DOE re-score comparison under contextual turnout-drop scoring
        # - HIL findings (keep competitive asymmetry, avoid boring lock-ish regimes,
        #   reduce over-penalization of asymmetric takeover with healthy entropy/competition)
        "ranges": {
            # Participation dynamics: centered on top-neighborhood with room for stress.
            "participation_init_q": (0.11, 0.18),
            "participation_alpha": (0.09, 0.155),
            "participation_beta": (3.9, 6.1),
            "election_cost_rate": (0.008, 0.032),
            "reward_rate_personal": (0.18, 0.29),
            "break_even_distance_common": (0.34, 0.62),

            # Puzzle/power regime knobs:
            # keep enough spread for seed-heterogeneous majority structure.
            "known_cells": (8.0, 16.0),
            "election_impact_on_mutation": (1.10, 2.95),
            "puzzle_local_kappa": (28.0, 100.0),
            "puzzle_shock_prob": (0.008, 0.070),
            "mu": (0.10, 0.41),

            # Satisfaction -> altruism mapping:
            # slightly sharper upper-slope region to test HIL "too dull" finding.
            "altruism_satisfaction_theta": (0.655, 0.755),
            "altruism_satisfaction_slope": (1.55, 3.45),
            "altruism_response_gamma": (0.88, 0.985),

            # Party learning signal shaping.
            "participation_signal_fee_weight": (0.42, 0.85),
            "participation_signal_group_shrink_k": (1.10, 5.80),
            "participation_signal_clip": (0.22, 0.45),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 250,
        },
    },
    "phase3_party_freeze_local_100seed_v1": {
        "name": "phase3_party_freeze_local_100seed_v1",
        # Final local freeze profile after supersearch:
        # - focus around top-performing regions from supersearch
        # - keep enough spread around seed-128/160 failure regimes
        # NOTE: run with --robust-every 1 so discriminability is observed for every design.
        "ranges": {
            # Participation dynamics (local around top-quantile region).
            "participation_init_q": (0.115, 0.165),
            "participation_alpha": (0.105, 0.150),
            "participation_beta": (4.40, 6.05),
            "election_cost_rate": (0.008, 0.029),
            "reward_rate_personal": (0.21, 0.285),
            "break_even_distance_common": (0.36, 0.60),

            # Puzzle / power regime:
            # known_cells kept high because top designs strongly concentrated there.
            "known_cells": (12.0, 18.0),
            "election_impact_on_mutation": (1.15, 2.55),
            "puzzle_local_kappa": (32.0, 90.0),
            "puzzle_shock_prob": (0.012, 0.065),
            "mu": (0.12, 0.34),

            # Satisfaction -> altruism mapping (local around best region).
            "altruism_satisfaction_theta": (0.66, 0.735),
            "altruism_satisfaction_slope": (1.75, 3.20),
            "altruism_response_gamma": (0.89, 0.975),

            # Party learning signal shaping.
            "participation_signal_fee_weight": (0.45, 0.75),
            "participation_signal_group_shrink_k": (1.30, 4.40),
            "participation_signal_clip": (0.24, 0.43),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 250,
        },
    },
    "phase3_party_altruism_sharp_probe_v1": {
        "name": "phase3_party_altruism_sharp_probe_v1",
        # Focused probe:
        # - keep most freeze-successful knobs tight
        # - deliberately open the unresolved deadlock drivers:
        #   altruism nonlinearity + puzzle motion + election-impact speed
        # - include known_cells because majority deadlock sensitivity is
        #   strongly affected by information breadth in recent analyses.
        "ranges": {
            # Keep stable participation/economy neighborhood mostly tight.
            "participation_init_q": (0.118, 0.162),
            "participation_alpha": (0.095, 0.150),
            "participation_beta": (4.55, 6.05),
            "election_cost_rate": (0.008, 0.029),
            "reward_rate_personal": (0.21, 0.285),
            "break_even_distance_common": (0.36, 0.60),

            # Keep puzzle/power family open where uncertainty remains.
            "known_cells": (10.0, 20.0),
            "election_impact_on_mutation": (0.95, 3.20),
            "puzzle_local_kappa": (18.0, 120.0),
            "puzzle_shock_prob": (0.002, 0.12),
            "mu": (0.10, 0.36),

            # Satisfaction -> altruism switch:
            # start from current upper-third slope and open to sharp regimes.
            "altruism_satisfaction_theta": (0.64, 0.84),
            "altruism_satisfaction_slope": (2.40, 10.0),
            "altruism_response_gamma": (0.90, 1.00),

            # Party learning signal shaping stays near validated region.
            "participation_signal_fee_weight": (0.45, 0.75),
            "participation_signal_group_shrink_k": (1.30, 4.40),
            "participation_signal_clip": (0.24, 0.45),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 250,
        },
    },
    "phase3_party_altruism_sharp_local_v1": {
        "name": "phase3_party_altruism_sharp_local_v1",
        # Follow-up local profile:
        # intended as a final medium local DOE before freeze.
        "ranges": {
            # Participation/economy: tighten around successful probe region.
            "participation_alpha": (0.108, 0.136),
            "participation_beta": (5.0, 5.4),
            "reward_rate_personal": (0.23, 0.26),
            "break_even_distance_common": (0.45, 0.55),

            # Keep knowledge conservative to preserve rule-difference signal
            # (avoid over-unifying altruistic voters).
            "known_cells": (11.0, 16.0),

            # Keep unresolved puzzle/power dynamics moderately open.
            "election_impact_on_mutation": (1.10, 3.10),
            "puzzle_local_kappa": (85.0, 125.0),
            "puzzle_shock_prob": (0.04, 0.08),
            "mu": (0.10, 0.30),

            # Satisfaction -> altruism switch:
            # - maintain sharp regime (higher slope)
            # - avoid very high theta that hurt viability in probe
            # - gamma<1 introduces delay/smoothing; keep somewhat open due uncertainty
            "altruism_satisfaction_theta": (0.64, 0.76),
            "altruism_satisfaction_slope": (4.50, 10.5),
            "altruism_response_gamma": (0.5, 1.0),

            # Party-learning shaping: tighten around validated neighborhood.
            "participation_signal_group_shrink_k": (1.60, 3.80),
            "participation_signal_clip": (0.27, 0.40),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "election_cost_rate": 0.02,
            "participation_signal_fee_weight": 0.6,
            "participation_signal_mode": "group_relative_delta_rel_party",
            # Freeze near the weighted center of top designs across the last
            # three party-focused DOEs.
            "participation_init_q": 0.14,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 250,
        },
    },
    "phase3_party_switch_puzzle_small_v1": {
        "name": "phase3_party_switch_puzzle_small_v1",
        # Small confirmation DOE:
        # freeze to design-0006-centered baseline and open only the
        # satisfaction-switch knobs + one puzzle-dominance knob.
        "ranges": {
            "altruism_satisfaction_theta": (0.70, 1.00),
            "altruism_satisfaction_slope": (6.0, 13.0),
            "altruism_response_gamma": (0.45, 0.95),
            # Direct puzzle-vs-power lever (knowledge coherence in altruistic voting).
            "known_cells": (16.0, 30.0),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_relative_delta_rel_party",
            # Design-0006-centered freeze values from DOE 20260302_175623.
            "participation_init_q": 0.14,
            "participation_alpha": 0.118717,
            "participation_beta": 5.041759,
            "election_cost_rate": 0.02,
            "reward_rate_personal": 0.231437,
            "break_even_distance_common": 0.459207,
            "known_cells": 16,
            "election_impact_on_mutation": 2.448928,
            "puzzle_local_kappa": 123.302102,
            "puzzle_shock_prob": 0.073023,
            "mu": 0.278457,
            "participation_signal_fee_weight": 0.6,
            "participation_signal_group_shrink_k": 1.954833,
            "participation_signal_clip": 0.277565,
        },
        "frozen_simulation": {
            **DEFAULT_FROZEN_SIM,
            "num_steps": 250,
        },
    },

}


def list_doe_profiles() -> tuple[str, ...]:
    return tuple(DEFAULT_DOE_PROFILES.keys())


def get_doe_profile(name: str) -> dict[str, Any]:
    key = str(name).strip()
    if key not in DEFAULT_DOE_PROFILES:
        raise ValueError(
            f"Unknown DOE profile: {name!r}. Available: {sorted(DEFAULT_DOE_PROFILES.keys())}"
        )
    return copy.deepcopy(DEFAULT_DOE_PROFILES[key])


@dataclass(frozen=True)
class DOERunTask:
    design_id: int
    seed: int
    rule_idx: int
    out_dir: Path
    params: dict[str, float]


def sample_design_points(
    *,
    num_points: int,
    rng,
    ranges: dict[str, tuple[float, float]] | None = None,
    max_tries_per_point: int = 2000,
) -> list[dict[str, float]]:
    """Sample DOE design points by uniform draws with algebra safety filtering."""
    if num_points <= 0:
        raise ValueError("num_points must be >= 1")
    r = DEFAULT_DOE_RANGES if ranges is None else ranges
    points: list[dict[str, float]] = []
    keys = sorted(r.keys())
    for _ in range(num_points):
        accepted = False
        for _try in range(max_tries_per_point):
            p: dict[str, float] = {}
            for k in keys:
                lo = float(r[k][0])
                hi = float(r[k][1])
                if k in INTEGER_DOE_KEYS:
                    p[k] = float(int(rng.integers(int(lo), int(hi) + 1)))
                else:
                    p[k] = float(rng.uniform(lo, hi))
            if _passes_rate_sum_constraint(p):
                points.append(p)
                accepted = True
                break
        if not accepted:
            raise RuntimeError("Could not sample a valid DOE point under constraints.")
    return points


def _passes_rate_sum_constraint(p: dict[str, float]) -> bool:
    if "election_cost_rate" not in p or "reward_rate_personal" not in p:
        return True
    return (
        float(p["election_cost_rate"])
        + float(p["reward_rate_personal"])
        <= 0.9
    )


def midpoint_params_from_ranges(ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Return midpoint parameter set for probe runs."""
    out: dict[str, float] = {}
    for k, (lo, hi) in ranges.items():
        out[k] = float((float(lo) + float(hi)) / 2.0)
    return out


def select_farthest_seeds_from_descriptors(
    descriptors: dict[int, np.ndarray],
    *,
    target_count: int,
) -> list[int]:
    """Greedy max-min farthest-point seed selection in standardized descriptor space."""
    if target_count <= 0:
        raise ValueError("target_count must be >= 1")
    if not descriptors:
        raise ValueError("descriptors must not be empty")
    seeds = sorted(int(s) for s in descriptors.keys())
    if target_count > len(seeds):
        raise ValueError(f"target_count={target_count} exceeds candidate seeds={len(seeds)}")

    vecs = [np.asarray(descriptors[s], dtype=float).reshape(-1) for s in seeds]
    dim = int(vecs[0].size)
    if any(int(v.size) != dim for v in vecs):
        raise ValueError("All descriptor vectors must have same dimensionality.")
    x = np.vstack(vecs)

    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma[sigma < 1e-12] = 1.0
    xz = (x - mu) / sigma

    centroid = np.mean(xz, axis=0)
    d0 = np.linalg.norm(xz - centroid, axis=1)
    first_idx = int(np.argmax(d0))
    selected_idx: list[int] = [first_idx]
    remaining: set[int] = set(range(len(seeds)))
    remaining.remove(first_idx)

    while len(selected_idx) < target_count:
        best_i = None
        best_val = -1.0
        for i in sorted(remaining):
            dist_to_sel = [float(np.linalg.norm(xz[i] - xz[j])) for j in selected_idx]
            score = float(min(dist_to_sel)) if dist_to_sel else 0.0
            if score > best_val + 1e-12:
                best_val = score
                best_i = i
        assert best_i is not None
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [seeds[i] for i in selected_idx]


def select_stratified_seeds(
    cfg,
    *,
    target_count: int,
    candidate_seeds: list[int],
    probe_rule_idx: int = 1,
    probe_params: dict[str, float] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    frozen_model: dict[str, Any] | None = None,
    frozen_simulation: dict[str, Any] | None = None,
) -> list[int]:
    """Select spread-out seeds based on initial-state descriptors from model instantiation."""
    uniq = sorted(set(int(s) for s in candidate_seeds))
    if target_count > len(uniq):
        raise ValueError(f"target_count={target_count} exceeds candidate seeds={len(uniq)}")
    r = DEFAULT_DOE_RANGES if ranges is None else ranges
    params = midpoint_params_from_ranges(r) if probe_params is None else probe_params

    # Local import to keep DOE utility module lightweight for non-selection paths.
    from src.model_setup import make_model

    descriptors: dict[int, np.ndarray] = {}
    for seed in uniq:
        cfg_probe = apply_doe_overrides(
            cfg,
            params=params,
            rule_idx=int(probe_rule_idx),
            base_seed=int(seed),
            frozen_model=frozen_model,
            frozen_simulation=frozen_simulation,
        )
        cfg_probe.model.seed = int(seed)
        model = make_model(cfg_probe.model, enable_datacollector=False)
        g = np.asarray(model.global_color_dst, dtype=float).reshape(-1)
        pg = np.asarray(model.personality_group_distribution, dtype=float).reshape(-1)
        desc = np.concatenate([g, pg]).astype(float)
        descriptors[int(seed)] = desc
    return select_farthest_seeds_from_descriptors(descriptors, target_count=target_count)


def write_seed_selection_manifest(
    *,
    out_root: Path,
    mode: str,
    selected_seeds: list[int],
    candidate_seeds: list[int] | None = None,
    probe_rule_idx: int | None = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seed_mode": str(mode),
        "selected_seeds": [int(s) for s in selected_seeds],
    }
    if candidate_seeds is not None:
        payload["candidate_seeds"] = [int(s) for s in candidate_seeds]
    if probe_rule_idx is not None:
        payload["probe_rule_idx"] = int(probe_rule_idx)
    (out_root / "doe_seed_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_run_plan(
    *,
    out_root: Path,
    design_points: list[dict[str, float]],
    seeds: list[int],
    primary_rule_idx: int = 1,
    robust_rule_idx: int = 2,
    include_robustness: bool = True,
    robust_every: int = 1,
) -> list[DOERunTask]:
    if robust_every <= 0:
        raise ValueError("robust_every must be >= 1")
    tasks: list[DOERunTask] = []
    for i, params in enumerate(design_points):
        design_dir = out_root / f"design_{i:04d}"
        for seed in seeds:
            pri_out = design_dir / f"rule_{rule_label(primary_rule_idx)}" / f"seed_{seed:05d}" / "run_0"
            tasks.append(
                DOERunTask(
                    design_id=i,
                    seed=int(seed),
                    rule_idx=int(primary_rule_idx),
                    out_dir=pri_out,
                    params=params,
                )
            )
            if include_robustness and (i % robust_every == 0):
                rob_out = design_dir / f"rule_{rule_label(robust_rule_idx)}" / f"seed_{seed:05d}" / "run_0"
                tasks.append(
                    DOERunTask(
                        design_id=i,
                        seed=int(seed),
                        rule_idx=int(robust_rule_idx),
                        out_dir=rob_out,
                        params=params,
                    )
                )
    return tasks


def apply_doe_overrides(
    cfg,
    *,
    params: dict[str, float],
    rule_idx: int,
    base_seed: int,
    frozen_model: dict[str, Any] | None = None,
    frozen_simulation: dict[str, Any] | None = None,
):
    """Return a deep-copied AppConfig with DOE overrides applied."""
    c = cfg.model_copy(deep=True)
    fm = DEFAULT_FROZEN_MODEL if frozen_model is None else frozen_model
    fs = DEFAULT_FROZEN_SIM if frozen_simulation is None else frozen_simulation
    for k, v in fm.items():
        setattr(c.model, k, v)
    for k, v in params.items():
        if k in INTEGER_DOE_KEYS:
            setattr(c.model, k, int(round(float(v))))
        else:
            setattr(c.model, k, float(v))
    c.model.rule_idx = int(rule_idx)
    for k, v in fs.items():
        setattr(c.simulation, k, v)
    c.simulation.base_seed = int(base_seed)
    return c


def rule_label(rule_idx: int) -> str:
    return RULE_LABELS.get(int(rule_idx), f"rule_{int(rule_idx)}")


def write_design_manifest(
    *,
    out_root: Path,
    design_points: list[dict[str, float]],
    seeds: list[int],
    primary_rule_idx: int,
    robust_rule_idx: int,
    include_robustness: bool,
    robust_every: int,
    ranges: dict[str, tuple[float, float]] | None = None,
    frozen_model: dict[str, Any] | None = None,
    frozen_simulation: dict[str, Any] | None = None,
    profile_name: str = "phase1",
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    r = DEFAULT_DOE_RANGES if ranges is None else ranges
    fm = DEFAULT_FROZEN_MODEL if frozen_model is None else frozen_model
    fs = DEFAULT_FROZEN_SIM if frozen_simulation is None else frozen_simulation
    spec = {
        "profile": str(profile_name),
        "design_points": len(design_points),
        "seeds": [int(s) for s in seeds],
        "primary_rule_idx": int(primary_rule_idx),
        "primary_rule_name": rule_label(primary_rule_idx),
        "robust_rule_idx": int(robust_rule_idx),
        "robust_rule_name": rule_label(robust_rule_idx),
        "include_robustness": bool(include_robustness),
        "robust_every": int(robust_every),
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in r.items()},
        "frozen_model": dict(fm),
        "frozen_simulation": dict(fs),
        "constraint": "election_cost_rate + reward_rate_personal <= 0.9",
    }
    (out_root / "doe_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    points_path = out_root / "doe_design_points.csv"
    keys = sorted(r.keys())
    with points_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["design_id", *keys])
        w.writeheader()
        for i, p in enumerate(design_points):
            row = {"design_id": i}
            for k in keys:
                row[k] = float(p[k])
            w.writerow(row)


def write_run_manifest(*, out_root: Path, plan: list[DOERunTask]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "doe_run_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "design_id",
                "seed",
                "rule_idx",
                "rule_name",
                "out_dir",
                "params_json",
                "params_hash",
            ],
        )
        w.writeheader()
        for i, task in enumerate(plan):
            params_json = json.dumps(task.params, sort_keys=True, separators=(",", ":"))
            params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
            w.writerow(
                {
                    "run_index": int(i),
                    "design_id": int(task.design_id),
                    "seed": int(task.seed),
                    "rule_idx": int(task.rule_idx),
                    "rule_name": rule_label(int(task.rule_idx)),
                    "out_dir": str(task.out_dir),
                    "params_json": params_json,
                    "params_hash": params_hash,
                }
            )
