"""
This file handles the definition of the canvas and model parameters.
"""
from typing import TYPE_CHECKING, cast
from mesa.visualization.modules import ChartModule
from src.agents.participation_agent import ColorCell
from src.participation_model import (
    ParticipationModel, distance_functions, social_welfare_functions
)
from math import factorial
from pathlib import Path
import mesa
import yaml
import os


def load_config(config_file=None):
    if config_file is None:
        config_file = os.environ.get("CONFIG_FILE", "config.yaml")
    config_path = Path(__file__).parent.parent / 'configs' / config_file
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf


# Load config
config = load_config()
cfg = config["model"]
vis_cfg = config.get("visualization", {})

# Colors
_COLORS = [
    "White", "Red", "Green", "Blue", "Yellow", "Aqua", "Fuchsia",
    "Lime", "Maroon", "Orange"
]  # 10 colors

def participation_draw(cell: ColorCell):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the cell in its current color.

    Args:
        cell: The cell in the simulation

    Returns:
        The portrayal dictionary.
    """
    if cell is None:
        raise AssertionError
    color = _COLORS[cell.color]
    draw_borders = vis_cfg.get("draw_borders", True)
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0,
                 "x": cell.row, "y": cell.col, "Color": color}
    # TODO: maybe: draw the agent number in the opposing color
    # If the cell is a border cell, change its appearance
    if TYPE_CHECKING:
        cell.model = cast(ParticipationModel, cell.model)
    if cell.is_border_cell and draw_borders:
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.9
        if color == "White":
            portrayal["Color"] = "LightGrey"
    # Add position (x, y) to the hover-text
    portrayal["Position"] = f"{cell.position}"
    portrayal["Color - text"] = _COLORS[cell.color]
    # Print number of agents in the cell if there are any
    if cell.num_agents_in_cell > 0:
        portrayal["text"] = str(cell.num_agents_in_cell)
        portrayal["text_color"] = "Black"
    for a in cell.areas:
        unique_id = a.unique_id
        if unique_id == -1:
            unique_id = "global"
        text = f"{a.num_agents} agents, color dist: {a.color_distribution}"
        portrayal[f"Area {unique_id}"] = text
    for voter in cell.agents:
        text = f"personality: {voter.personality}, assets: {voter.assets}"
        portrayal[f"Agent {voter.unique_id}"] = text
    return portrayal

canvas_element = mesa.visualization.CanvasGrid(
    participation_draw,
    cfg["width"],
    cfg["height"],
    cfg["width"] * vis_cfg["cell_size"],
    cfg["height"] * vis_cfg["cell_size"]
)

wealth_chart = ChartModule(
    [{"Label": "Collective assets", "Color": "Black"}],
    data_collector_name='datacollector'
)

color_distribution_chart = ChartModule(
    [{"Label": f"Color {i}",
      "Color": "LightGrey" if _COLORS[i] == "White" else _COLORS[i]}
     for i in range(len(_COLORS))],
    data_collector_name='datacollector'
)

voter_turnout = ChartModule(
    [{"Label": "Voter turnout globally (in percent)", "Color": "Black"},
     {"Label": "Gini Index (0-100)", "Color": "Red"}],
    data_collector_name='datacollector'
)

visualization_params = {
    "draw_borders": mesa.visualization.Checkbox(
            name="Draw border cells", value=vis_cfg.get("draw_borders", True)
        ),
    "show_area_stats": mesa.visualization.Checkbox(
        name="Show all statistics", value=cfg.get("show_area_stats", True)
    ),
}

model_params = {
    "height": cfg["height"],
    "width": cfg["width"],
    "rule_idx": mesa.visualization.Slider(
        name=f"Rule index {[r.__name__ for r in social_welfare_functions]}",
        value=cfg["rule_idx"], min_value=0, max_value=len(social_welfare_functions)-1,
    ),
    "distance_idx": mesa.visualization.Slider(
        name=f"Dist-Function index {[f.__name__ for f in distance_functions]}",
        value=cfg["distance_idx"], min_value=0, max_value=len(distance_functions)-1,
    ),
    "election_costs": mesa.visualization.Slider(
        name="Election costs", value=cfg["election_costs"], min_value=0, max_value=100,
        step=1, description="The costs for participating in an election"
    ),
    "max_reward": mesa.visualization.Slider(
        name="Maximal reward", value=cfg["max_reward"], min_value=0,
        max_value=cfg["election_costs"]*100,
        step=1, description="The costs for participating in an election"
    ),
    "mu": mesa.visualization.Slider(
        name="Mutation rate", value=cfg["mu"], min_value=0.001, max_value=0.5,
        step=0.001, description="Probability of a color cell to mutate"
    ),
    "election_impact_on_mutation": mesa.visualization.Slider(
        name="Election impact on mutation", value=cfg["election_impact_on_mutation"],
        min_value=0.1, max_value=5.0, step=0.1,
        description="Factor determining how strong mutation accords to election"
    ),
    "num_agents": mesa.visualization.Slider(
        name="# Agents", value=cfg["num_agents"], min_value=10, max_value=99999,
        step=10
    ),
    "num_colors": mesa.visualization.Slider(
        name="# Colors", value=cfg["num_colors"], min_value=2, max_value=len(_COLORS),
        step=1
    ),
    "num_personalities": mesa.visualization.Slider(
        name="# different personalities", value=cfg["num_personalities"],
        min_value=1, max_value=factorial(cfg["num_colors"]), step=1
    ),
    "common_assets": mesa.visualization.Slider(
        name="Initial common assets", value=cfg["common_assets"],
        min_value=cfg["num_agents"], max_value=1000*cfg["num_agents"], step=10
    ),
    "known_cells": mesa.visualization.Slider(
        name="# known fields", value=cfg["known_cells"],
        min_value=1, max_value=100, step=1
    ),
    "color_patches_steps": mesa.visualization.Slider(
        name="Patches size (# steps)", value=cfg["color_patches_steps"],
        min_value=0, max_value=9, step=1,
        description="More steps lead to bigger color patches"
    ),
    "patch_power": mesa.visualization.Slider(
        name="Patches power", value=cfg["patch_power"], min_value=0.0, max_value=3.0,
        step=0.2, description="Increases the power/radius of the color patches"
    ),
    "heterogeneity": mesa.visualization.Slider(
        name="Global color distribution heterogeneity",
        value=cfg["heterogeneity"], min_value=0.0, max_value=0.9, step=0.1,
        description="The higher the heterogeneity factor the greater the" +
                    "difference in how often some colors appear overall"
    ),
    "num_areas": mesa.visualization.Slider(
        name=f"# Areas within the {cfg['height']}x{cfg['width']} world", step=1,
        value=cfg["num_areas"], min_value=1, max_value=min(cfg["width"], cfg["height"])//2
    ),
    "av_area_height": mesa.visualization.Slider(
        name="Av. area height", value=cfg["av_area_height"],
        min_value=2, max_value=cfg["height"]//2,
        step=1, description="Select the average height of an area"
    ),
    "av_area_width": mesa.visualization.Slider(
        name="Av. area width", value=cfg["av_area_width"],
        min_value=2, max_value=cfg["width"]//2,
        step=1, description="Select the average width of an area"
    ),
    "area_size_variance": mesa.visualization.Slider(
        name="Area size variance", value=cfg["area_size_variance"],
        # TODO there is a division by zero error for value=1.0 - check this
        min_value=0.0, max_value=0.99, step=0.1,
        description="Select the variance of the area sizes"
    ),
}
