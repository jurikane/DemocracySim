"""
handles the definition of the canvas parameters and
the drawing of the model representation on the canvas
"""
# import webbrowser
import mesa
from mesa.visualization.modules import ChartModule
from participation_model import ParticipationModel
from participation_agent import ColorCell, VoteAgent

_COLORS = [
    "White",
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Aqua",
    "Fuchsia",
    "Gray",
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

    :param cell:  the cell in the simulation

    :return: the portrayal dictionary.
    """
    if cell is None:
        raise AssertionError
    if isinstance(cell, VoteAgent):
        return None
    # # Retrieve the agents of the cell
    # agents = cell.model.grid.get_cell_list_contents([cell.pos])
    # # Count the number of ParticipationAgents (subtracting the color cell)
    # nr_agents = len(agents)
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0,
                 "x": cell.row, "y": cell.col,
                 "Color": _COLORS[cell.color]}
    if cell.num_agents_in_cell > 0:
        portrayal["text"] = str(cell.num_agents_in_cell)
        portrayal["text_color"] = "Black"
    return portrayal


grid_rows = 80
grid_cols = 100
cell_size = 10
canvas_width = grid_rows * cell_size
canvas_height = grid_cols * cell_size

canvas_element = mesa.visualization.CanvasGrid(
    participation_draw, grid_rows, grid_cols, canvas_width, canvas_height
)

happy_chart = mesa.visualization.ChartModule([{"Label": "happy",
                                               "Color": "Black"}])


def compute_wealth(model: ParticipationModel):
    agents_wealth = [agent.wealth for agent in model.agent_scheduler.agents]
    return {"wealth": agents_wealth}


wealth_chart = mesa.visualization.modules.ChartModule(
    [{"Label": "wealth", "Color": "Black"}],
    data_collector_name='datacollector'
)

model_params = {
    "height": mesa.visualization.Slider(
        name="World Height", value=grid_cols, min_value=10, max_value=1000,
        step=10, description="Select the height of the world"
    ),
    "width": mesa.visualization.Slider(
        name="World Width", value=grid_rows, min_value=10, max_value=1000,
        step=10, description="Select the width of the world"
    ),
    "num_agents": mesa.visualization.Slider(
        name="# Agents", value=800, min_value=10, max_value=99999, step=10
    ),
    "num_colors": mesa.visualization.Slider(
        name="# Colors", value=4, min_value=2, max_value=len(_COLORS), step=1
    ),
    # "num_regions": mesa.visualization.Slider(
    #     name="# Regions", value=4, min_value=4, max_value=500, step=1
    # ),
}
