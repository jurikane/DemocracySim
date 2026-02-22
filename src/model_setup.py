"""
Wires config -> ParticipationModel kwargs -> Mesa UI server.
"""

from src.config.schema import AppConfig, ModelConfig
from math import factorial
import mesa
from mesa.visualization.ModularVisualization import ModularServer
from src.models.participation_model import (
    ParticipationModel,
    distance_functions,
    social_welfare_functions,
    social_welfare_function_short_names,
    distance_function_short_names
)
from src.viz.factory import make_canvas, make_charts

# The arguments accepted by ParticipationModel.__init__
_ALLOWED_KW = {
    "height",
    "width",
    "num_agents",
    "num_colors",
    "num_personality_groups",
    "mu",
    "election_impact_on_mutation",
    "initial_agent_assets",
    "known_cells",
    "num_areas",
    "av_area_height",
    "av_area_width",
    "area_size_variance",
    "patch_power",
    "color_patches_steps",
    "heterogeneity",
    "rule_idx",
    "distance_idx",
    "election_cost_rate",
    "reward_rate_personal",
    "break_even_distance_common",
    "quality_target_mode",
    "puzzle_local_kappa",
    "puzzle_shock_prob",
    "seed",
    # Adaptive participation learning
    "participation_alpha",
    "participation_beta",
    "participation_init_q",
    "participation_q_max",
    "bias_toward_participation",
    "participation_baseline_alpha",
    # Adaptive altruism learning
    "altruism_alpha",
    "altruism_init",
    "altruism_clip_min",
    "altruism_clip_max",
    "altruism_mode",
    "altruism_response_gamma",
    "altruism_learning",
    "altruism_static",
    "satisfaction_mode",
    "satisfaction_baseline_alpha",
    # Per-agent personal preference distribution shape
    "personal_preference_peakedness",
}


def build_model_kwargs(model_cfg: ModelConfig) -> dict:
    """
    Create a kwargs dict for ParticipationModel from a config mapping
    (used for headless or non-interactive runs).
    """
    return {k: getattr(model_cfg, k) for k in _ALLOWED_KW if
            hasattr(model_cfg, k)}


def build_model_params(model_cfg: ModelConfig) -> dict:
    """
    Create Mesa UI sliders/params so the web UI shows controls.

    """
    params = {}
    if model_cfg.seed is not None:
        params["seed"] = mesa.visualization.Slider(
            name="Seed (None = random)",
            value=model_cfg.seed,
            min_value=0,
            max_value=200,
            step=1,
        )
    # Add the rest of the params (except seed, which is optional) as sliders
    params.update({
        "height": model_cfg.height,
        "width": model_cfg.width,
        "rule_idx": mesa.visualization.Slider(
            name=f"Rule {social_welfare_function_short_names}",
            value=model_cfg.rule_idx,
            min_value=0,
            max_value=len(social_welfare_functions) - 1,
            step=1,
        ),
        "distance_idx": mesa.visualization.Slider(
            name=f"Dist {distance_function_short_names}",
            value=model_cfg.distance_idx,
            min_value=0,
            max_value=len(distance_functions) - 1,
            step=1,
        ),
        "election_cost_rate": mesa.visualization.Slider(
            name="Vote-Cost/Effort rate (wealth-scaled)",
            value=model_cfg.election_cost_rate,
            min_value=0,
            max_value=1,
            step=0.01,
        ),
        "reward_rate_personal": mesa.visualization.Slider(
            name="Reward/punishment rate",
            value=model_cfg.reward_rate_personal,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "break_even_distance_common": mesa.visualization.Slider(
            name="Quality threshold (dist_to_reality)",
            value=model_cfg.break_even_distance_common,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "quality_target_mode": mesa.visualization.Slider(
            name="Quality target mode (0=reality, 1=puzzle)",
            value=1 if str(getattr(model_cfg, "quality_target_mode", "puzzle")) == "puzzle" else 0,
            min_value=0,
            max_value=1,
            step=1,
        ),
        "puzzle_local_kappa": mesa.visualization.Slider(
            name="Puzzle local kappa (higher=smaller jumps)",
            value=float(getattr(model_cfg, "puzzle_local_kappa", 30.0)),
            min_value=0.1,
            max_value=200.0,
            step=0.1,
        ),
        "puzzle_shock_prob": mesa.visualization.Slider(
            name="Puzzle shock probability",
            value=float(getattr(model_cfg, "puzzle_shock_prob", 0.05)),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "participation_alpha": mesa.visualization.Slider(
            name="Participation learning alpha",
            value=model_cfg.participation_alpha,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "participation_beta": mesa.visualization.Slider(
            name="Participation sigmoid beta",
            value=model_cfg.participation_beta,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        ),
        "participation_init_q": mesa.visualization.Slider(
            name="Participation init q",
            value=model_cfg.participation_init_q,
            min_value=-5.0,
            max_value=5.0,
            step=0.1,
        ),
        "participation_q_max": mesa.visualization.Slider(
            name="Participation q clip max",
            value=model_cfg.participation_q_max,
            min_value=0.0,
            max_value=5.0,
            step=0.1,
        ),
        "bias_toward_participation": mesa.visualization.Slider(
            name="Bias toward participation",
            value=model_cfg.bias_toward_participation,
            min_value=0.0,
            max_value=0.5,
            step=0.01,
        ),
        "participation_baseline_alpha": mesa.visualization.Slider(
            name="Participation baseline alpha (EMA)",
            value=model_cfg.participation_baseline_alpha,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "altruism_learning": mesa.visualization.Slider(
            name="Altruism learning (0/1)",
            value=int(bool(model_cfg.altruism_learning)),
            min_value=0,
            max_value=1,
            step=1,
        ),
        "altruism_static": mesa.visualization.Slider(
            name="Altruism static (used when learning=0)",
            value=model_cfg.altruism_static,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "altruism_response_gamma": mesa.visualization.Slider(
            name="Altruism response gamma (satisfaction mode)",
            value=float(getattr(model_cfg, "altruism_response_gamma", 1.0)),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "altruism_mode": mesa.visualization.Slider(
            name="Altruism mode (0=static, 1=surprise, 2=satisfaction)",
            value={"static": 0, "surprise_learning": 1, "satisfaction": 2}.get(
                str(getattr(model_cfg, "altruism_mode", "satisfaction")),
                2,
            ),
            min_value=0,
            max_value=2,
            step=1,
        ),
        "altruism_alpha": mesa.visualization.Slider(
            name="Altruism learning alpha",
            value=model_cfg.altruism_alpha,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "altruism_init": mesa.visualization.Slider(
            name="Altruism init factor",
            value=model_cfg.altruism_init,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "satisfaction_baseline_alpha": mesa.visualization.Slider(
            name="Satisfaction baseline alpha (EMA)",
            value=model_cfg.satisfaction_baseline_alpha,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
        "num_agents": mesa.visualization.Slider(
            name="# Agents",
            value=model_cfg.num_agents,
            min_value=10,
            max_value=1500,
            step=10,
        ),
        "num_colors": mesa.visualization.Slider(
            name="# Colors",
            value=model_cfg.num_colors,
            min_value=2,
            max_value=8,
            step=1,
        ),
        "num_personality_groups": mesa.visualization.Slider(
            name="# different personality_groups",
            value=model_cfg.num_personality_groups,
            min_value=1,
            max_value=max(1, factorial(model_cfg.num_colors)),
            step=1,
        ),
        "known_cells": mesa.visualization.Slider(
            name="Number of known cells per agent",
            value=model_cfg.known_cells,
            min_value=1,
            max_value=100,
            step=1,
        ),
        # Optional UI knob for robustness checks (kept off by default).
        # "initial_agent_assets": mesa.visualization.Slider(
        #     name="Initial assets per agent",
        #     value=model_cfg.initial_agent_assets,
        #     min_value=0.0,
        #     max_value=1000.0,
        #     step=1.0,
        # ),
        "mu": mesa.visualization.Slider(
            name="Mutation rate",
            value=model_cfg.mu,
            min_value=0.001,
            max_value=1.0,
            step=0.001,
        ),
        "election_impact_on_mutation": mesa.visualization.Slider(
            name="Election impact on mutation",
            value=model_cfg.election_impact_on_mutation,
            min_value=0.1,
            max_value=5.0,
            step=0.1,
        ),
        "personal_preference_peakedness": mesa.visualization.Slider(
            name="Personal preference peakedness",
            value=model_cfg.personal_preference_peakedness,
            min_value=0.1,
            max_value=5.0,
            step=0.1,
        ),
        "color_patches_steps": mesa.visualization.Slider(
            name="Patches size (# steps)",
            value=model_cfg.color_patches_steps,
            min_value=0,
            max_value=9,
            step=1,
        ),
        "patch_power": mesa.visualization.Slider(
            name="Patches power",
            value=model_cfg.patch_power,
            min_value=0.0,
            max_value=3.0,
            step=0.2,
        ),
        "heterogeneity": mesa.visualization.Slider(
            name="Global color distribution heterogeneity",
            value=model_cfg.heterogeneity,
            min_value=0.0,
            max_value=0.9,
            step=0.1,
        ),
        "num_areas": mesa.visualization.Slider(
            name=f"# Areas within the {model_cfg.height}x{model_cfg.width} world",
            value=model_cfg.num_areas,
            min_value=1,
            max_value=max(1, min(model_cfg.width, model_cfg.height) // 2),
            step=1,
        ),
        "av_area_height": mesa.visualization.Slider(
            name="Av. area height",
            value=model_cfg.av_area_height,
            min_value=2,
            max_value=max(2, model_cfg.height // 2),
            step=1,
        ),
        "av_area_width": mesa.visualization.Slider(
            name="Av. area width",
            value=model_cfg.av_area_width,
            min_value=2,
            max_value=max(2, model_cfg.width // 2),
            step=1,
        ),
        "area_size_variance": mesa.visualization.Slider(
            name="Area size variance",
            value=model_cfg.area_size_variance,
            min_value=0.0,
            max_value=0.99,
            step=0.1,
        ),
    })
    return params


def make_model(model_cfg: ModelConfig, *, enable_datacollector: bool = True) -> ParticipationModel:
    """
    Instantiate the model using the loaded config (non-UI usage).
    """
    kwargs = build_model_kwargs(model_cfg)
    kwargs["enable_datacollector"] = bool(enable_datacollector)
    return ParticipationModel(**kwargs)


def make_server(cfg: AppConfig) -> ModularServer:
    """
    Build the ModularServer with CanvasGrid, charts, and UI sliders.
    """
    vis_cfg = cfg.visualization
    elements = [make_canvas(cfg), *make_charts(cfg)]
    title = getattr(vis_cfg, "title", "Participation Model")

    # Use interactive model parameters (sliders appear in the UI)
    params = build_model_params(cfg.model)

    return ModularServer(ParticipationModel, elements, title, params)
