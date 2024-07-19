"""
handles the definition of the canvas parameters and
the drawing of the model representation on the canvas
"""
from typing import TYPE_CHECKING, cast
# import webbrowser
import mesa
from mesa.visualization.modules import ChartModule
from participation_model import ParticipationModel
from participation_agent import ColorCell, VoteAgent
import matplotlib.pyplot as plt


# Model grid parameters
grid_rows = 100  # height
grid_cols = 80  # width
cell_size = 10
canvas_height = grid_rows * cell_size
canvas_width = grid_cols * cell_size
draw_borders = True
# Colors and agents
num_colors = 4
color_adj_steps = 3
color_heterogeneity = 0.3
num_agents = 800
# Voting area parameters
# num_areas = 25
# area_height = 20  # grid_rows // int(sqrt(num_areas))
# area_width = 16  # grid_cols // int(sqrt(num_areas))
num_areas = 16
area_height = 25
area_width = 20
area_var = 0.0
# Voting rules TODO: Implement voting rules
voting_rules = ["Majority Rule", "Approval Voting", "Kemeny"]
rule_idx = 0


_COLORS = [
    "White",
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Aqua",
    "Fuchsia",
    "Lavender",
    "Lime",
    "Maroon",
    "Navy",
    "Olive",
    "Orange",
    "Purple",
    "Silver",
    "Teal",
    "Pink",
    "Brown",
    "Gold",
    "Coral",
    "Crimson",
    "DarkBlue",
    "DarkRed",
    "DarkGreen",
    "DarkKhaki",
    "DarkMagenta",
    "DarkOliveGreen",
    "DarkOrange",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
]  # 30 colors


def participation_draw(cell: ColorCell):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the cell in its current color.

    :param cell: The cell in the simulation

    :return: The portrayal dictionary.
    """
    if cell is None:
        raise AssertionError
    if isinstance(cell, VoteAgent):
        return None
    color = _COLORS[cell.color]
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0,
                 "x": cell.row, "y": cell.col,
                 "Color": color}
    # TODO: add the areas the cell belongs to to the hover-text (the text that is shown when one hovers over the cell in the grid)
    #       + maybe: draw the agent number in the opposing color, + maybe draw borders nicer
    # If the cell is a border cell, change its appearance
    if TYPE_CHECKING:  # Type hint for IDEs
        cell.model = cast(ParticipationModel, cell.model)
    if cell.is_border_cell and cell.model.draw_borders:
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.9  # Adjust the radius to fit within the cell
        if color == "White":
            portrayal["Color"] = "Grey"
    if cell.num_agents_in_cell > 0:
        portrayal["text"] = str(cell.num_agents_in_cell)
        portrayal["text_color"] = "Black"
    return portrayal


canvas_element = mesa.visualization.CanvasGrid(
    participation_draw, grid_cols, grid_rows, canvas_width, canvas_height
)


# Draw bars TODO: Implement to use within the mesa framework..
def draw_color_dist_bars(color_distributions):
    # Setup plot
    fig, ax = plt.subplots()
    for i, dist in enumerate(color_distributions):
        bottom = 0
        for j, part in enumerate(color_distributions):
            ax.bar(i, part, bottom=bottom, color=_COLORS[j % len(_COLORS)])
            bottom += part
    # Set x-ticks to be distribution indices
    plt.xticks(range(len(color_distributions)))
    plt.show()


a_chart = mesa.visualization.ChartModule([{"Label": "Number of agents",
                                           "Color": "Black"}],
                                         data_collector_name='datacollector')


wealth_chart = mesa.visualization.modules.ChartModule(
    [{"Label": "Collective assets", "Color": "Black"}],
    data_collector_name='datacollector'
)

model_params = {
    "height": grid_rows,
    "width": grid_cols,
    # "voting_rule": mesa.visualization.modules.DropDown(
    #     name="Voting Rule",
    #     options=voting_rules,
    #     value=voting_rules[0],  # Default value
    # ),
    "voting_rule": mesa.visualization.Slider(
        name=f"Rule index {voting_rules}", value=rule_idx,
        min_value=0, max_value=2,
    ),
    "draw_borders": mesa.visualization.Checkbox(
        name="Draw border cells", value=draw_borders
    ),
    "num_agents": mesa.visualization.Slider(
        name="# Agents", value=num_agents, min_value=10, max_value=99999,
        step=10
    ),
    "num_colors": mesa.visualization.Slider(
        name="# Colors", value=num_colors, min_value=2, max_value=len(_COLORS),
        step=1
    ),
    "color_patches_steps": mesa.visualization.Slider(
        name="Patches size (# steps)", value=color_adj_steps,
        min_value=0, max_value=9, step=1,
        description="More steps lead to bigger color patches"
    ),
    "patch_power": mesa.visualization.Slider(
        name="Patches power", value=1.0, min_value=0.0, max_value=3.0,
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
        name="Av. area height", value=area_height,
        min_value=2, max_value=grid_rows//2,
        step=1, description="Select the average height of an area"
    ),
    "av_area_width": mesa.visualization.Slider(
        name="Av. area width", value=area_width,
        min_value=2, max_value=grid_cols//2,
        step=1, description="Select the average width of an area"
    ),
    "area_size_variance": mesa.visualization.Slider(
        name="Area size variance", value=area_var, min_value=0.0, max_value=1.0,
        step=0.1, description="Select the variance of the area sizes"
    ),
}
