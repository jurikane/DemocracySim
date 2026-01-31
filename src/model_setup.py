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
)
from src.viz.factory import make_canvas, make_charts

# The arguments accepted by ParticipationModel.__init__
_ALLOWED_KW = {
    "height",
    "width",
    "num_agents",
    "num_colors",
    "num_personalities",
    "mu",
    "election_impact_on_mutation",
    "common_assets",
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
    "election_costs",
    "seed",
    # Adaptive participation learning
    "participation_alpha",
    "participation_beta",
    "participation_init_q",
    "participation_q_max",
    # Per-agent personal_opt_dist
    "personal_opt_dist_concentration",
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
    params = {
        "height": model_cfg.height,
        "width": model_cfg.width,
        "seed": getattr(model_cfg, "seed", None),  # Optional seed
        "rule_idx": mesa.visualization.Slider(
            name=f"Rule index {[r.__name__ for r in social_welfare_functions]}",
            value=model_cfg.rule_idx,
            min_value=0,
            max_value=len(social_welfare_functions) - 1,
            step=1,
        ),
        "distance_idx": mesa.visualization.Slider(
            name=f"Dist-Function index {[f.__name__ for f in distance_functions]}",
            value=model_cfg.distance_idx,
            min_value=0,
            max_value=len(distance_functions) - 1,
            step=1,
        ),
        "election_costs": mesa.visualization.Slider(
            name="Election costs in %",
            value=model_cfg.election_costs,
            min_value=0,
            max_value=1,
            step=0.01,
        ),
        "mu": mesa.visualization.Slider(
            name="Mutation rate",
            value=model_cfg.mu,
            min_value=0.001,
            max_value=0.5,
            step=0.001,
        ),
        "election_impact_on_mutation": mesa.visualization.Slider(
            name="Election impact on mutation",
            value=model_cfg.election_impact_on_mutation,
            min_value=0.1,
            max_value=5.0,
            step=0.1,
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
            max_value=max(2, model_cfg.num_colors),
            step=1,
        ),
        "num_personalities": mesa.visualization.Slider(
            name="# different personalities",
            value=model_cfg.num_personalities,
            min_value=1,
            max_value=max(1, factorial(model_cfg.num_colors)),
            step=1,
        ),
        "common_assets": mesa.visualization.Slider(
            name="Initial common assets",
            value=model_cfg.common_assets,
            min_value=model_cfg.num_agents,
            max_value=1000 * model_cfg.num_agents,
            step=10,
        ),
        "known_cells": mesa.visualization.Slider(
            name="# known fields",
            value=model_cfg.known_cells,
            min_value=1,
            max_value=100,
            step=1,
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
    }
    return params


def make_model(model_cfg: ModelConfig) -> ParticipationModel:
    """
    Instantiate the model using the loaded config (non-UI usage).
    """
    kwargs = build_model_kwargs(model_cfg)
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
