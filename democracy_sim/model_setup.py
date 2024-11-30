"""
This file handles the definition of the canvas and model parameters.
"""
from typing import TYPE_CHECKING, cast
from mesa.visualization.modules import ChartModule
from democracy_sim.participation_agent import ColorCell
from democracy_sim.participation_model import (ParticipationModel,
                                               distance_functions,
                                               social_welfare_functions)
from math import factorial
import mesa

# Parameters

#############
# Elections #
#############
election_costs = 5
max_reward = 50
election_impact_on_mutation = 1.0  # 0.1-5.0
mu = 0.01  # 0.001-0.5
# Voting rules (see social_welfare_functions.py)
rule_idx = 0
# Distance functions (see distance_functions.py)
distance_idx = 1
####################
# Model parameters #
####################
num_agents = 800
# Colors
num_colors = 4
color_patches_steps = 3
patch_power = 1.0
color_heterogeneity = 0.3
# Voting Agents
num_personality_colors = 4  # TODO: does this make sense when we have to use orderings anyways?
num_personalities = 10
# Grid
grid_rows = 100  # height
grid_cols = 80  # width
cell_size = 10
canvas_height = grid_rows * cell_size
canvas_width = grid_cols * cell_size
draw_borders = True
# Voting Areas
# num_areas = 16
# av_area_height = 25
# # area_height = grid_rows // int(sqrt(num_areas))
# av_area_width = 20
# # area_width = grid_cols // int(sqrt(num_areas))
num_areas = 4
av_area_height = 50
av_area_width = 40
area_size_variance = 0.0
########################
# Statistics and Views #
########################
show_area_stats = True


_COLORS = [
    "White",
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Aqua",
    "Fuchsia",
    #"Lavender",
    "Lime",
    "Maroon",
    #"Navy",
    #"Olive",
    "Orange",
    #"Purple",
    #"Silver",
    #"Teal",
    # "Pink",
    # "Brown",
    # "Gold",
    # "Coral",
    # "Crimson",
    # "DarkBlue",
    # "DarkRed",
    # "DarkGreen",
    # "DarkKhaki",
    # "DarkMagenta",
    # "DarkOliveGreen",
    # "DarkOrange",
    # "DarkTurquoise",
    # "DarkViolet",
    # "DeepPink",
]  # 10 colors


def participation_draw(cell: ColorCell):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the cell in its current color.

    :param cell: The cell in the simulation

    :return: The portrayal dictionary.
    """
    if cell is None:
        raise AssertionError
    color = _COLORS[cell.color]
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0,
                 "x": cell.row, "y": cell.col,
                 "Color": color}
    # TODO: maybe: draw the agent number in the opposing color
    # If the cell is a border cell, change its appearance
    if TYPE_CHECKING:  # Type hint for IDEs
        cell.model = cast(ParticipationModel, cell.model)
    if cell.is_border_cell and cell.model.draw_borders:
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.9  # Adjust the radius to fit within the cell
        if color == "White":
            portrayal["Color"] = "LightGrey"
    # Add position (x, y) to the hover-text
    portrayal["Position"] = f"{cell.position}"
    portrayal["Color - text"] = _COLORS[cell.color]
    if cell.num_agents_in_cell > 0:
        portrayal[f"text"] = str(cell.num_agents_in_cell)
        portrayal["text_color"] = "Black"
    for a in cell.areas:
        unique_id = a.unique_id
        if unique_id == -1:
            unique_id = "global"
        text = f"{a.num_agents} agents, color dist: {a.color_distribution}"
        portrayal[f"Area {unique_id}"] = text
    for voter in cell.agents:
        text = f"personality: {voter.personality}"
        portrayal[f"Agent {voter.unique_id}"] = text
    return portrayal


canvas_element = mesa.visualization.CanvasGrid(
    participation_draw, grid_cols, grid_rows, canvas_width, canvas_height
)


wealth_chart = mesa.visualization.modules.ChartModule(
    [{"Label": "Collective assets", "Color": "Black"}],
    data_collector_name='datacollector'
)


color_distribution_chart = mesa.visualization.modules.ChartModule(
        [{"Label": f"Color {i}",
          "Color": "LightGrey" if _COLORS[i] == "White" else _COLORS[i]}
         for i in range(len(_COLORS))],
        data_collector_name='datacollector'
    )

voter_turnout = mesa.visualization.ChartModule(
    [{"Label": "Voter turnout globally (in percent)", "Color": "Black"},
     {"Label": "Gini Index (0-100)", "Color": "Red"}],
    data_collector_name='datacollector')


model_params = {
    "height": grid_rows,
    "width": grid_cols,
    "draw_borders": mesa.visualization.Checkbox(
        name="Draw border cells", value=draw_borders
    ),
    "rule_idx": mesa.visualization.Slider(
        name=f"Rule index {[r.__name__ for r in social_welfare_functions]}",
        value=rule_idx, min_value=0, max_value=len(social_welfare_functions)-1,
    ),
    "distance_idx": mesa.visualization.Slider(
        name=f"Dist-Function index {[f.__name__ for f in distance_functions]}",
        value=distance_idx, min_value=0, max_value=len(distance_functions)-1,
    ),
    "election_costs": mesa.visualization.Slider(
        name="Election costs", value=election_costs, min_value=0, max_value=100,
        step=1, description="The costs for participating in an election"
    ),
    "max_reward": mesa.visualization.Slider(
        name="Maximal reward", value=max_reward, min_value=0,
        max_value=election_costs*100,
        step=1, description="The costs for participating in an election"
    ),
    "mu": mesa.visualization.Slider(
        name="Mutation rate", value=mu, min_value=0.001, max_value=0.5,
        step=0.001, description="Probability of a color cell to mutate"
    ),
    "election_impact_on_mutation": mesa.visualization.Slider(
        name="Election impact on mutation", value=election_impact_on_mutation,
        min_value=0.1, max_value=5.0, step=0.1,
        description="Factor determining how strong mutation accords to election"
    ),
    "num_agents": mesa.visualization.Slider(
        name="# Agents", value=num_agents, min_value=10, max_value=99999,
        step=10
    ),
    "num_colors": mesa.visualization.Slider(
        name="# Colors", value=num_colors, min_value=2, max_value=len(_COLORS),
        step=1
    ),
    "num_personalities": mesa.visualization.Slider(
        name="# of different personalities", value=num_personalities,
        min_value=1, max_value=factorial(num_personality_colors), step=1
    ),
    "num_personality_colors": mesa.visualization.Slider(
        name="# colors determining the personality",
        value=num_personality_colors,
        min_value=1, max_value=num_colors-1, step=1
    ),
    "color_patches_steps": mesa.visualization.Slider(
        name="Patches size (# steps)", value=color_patches_steps,
        min_value=0, max_value=9, step=1,
        description="More steps lead to bigger color patches"
    ),
    "patch_power": mesa.visualization.Slider(
        name="Patches power", value=patch_power, min_value=0.0, max_value=3.0,
        step=0.2, description="Increases the power/radius of the color patches"
    ),
    "heterogeneity": mesa.visualization.Slider(
        name="Global color distribution heterogeneity",
        value=color_heterogeneity, min_value=0.0, max_value=0.9, step=0.1,
        description="The higher the heterogeneity factor the greater the" +
                    "difference in how often some colors appear overall"
    ),
    "num_areas": mesa.visualization.Slider(
        name=f"# Areas within the {grid_rows}x{grid_cols} world", step=1,
        value=num_areas, min_value=4, max_value=min(grid_cols, grid_rows)//2
    ),
    "av_area_height": mesa.visualization.Slider(
        name="Av. area height", value=av_area_height,
        min_value=2, max_value=grid_rows//2,
        step=1, description="Select the average height of an area"
    ),
    "av_area_width": mesa.visualization.Slider(
        name="Av. area width", value=av_area_width,
        min_value=2, max_value=grid_cols//2,
        step=1, description="Select the average width of an area"
    ),
    "area_size_variance": mesa.visualization.Slider(
        name="Area size variance", value=area_size_variance,
        # TODO there is a division by zero error for value=1.0 - check this
        min_value=0.0, max_value=0.99, step=0.1,
        description="Select the variance of the area sizes"
    ),
    "show_area_stats": mesa.visualization.Checkbox(
            name="Show all statistics", value=show_area_stats
        ),
}
