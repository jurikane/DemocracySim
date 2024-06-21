"""
handles the definition of the canvas parameters and
the drawing of the model representation on the canvas
"""
# import webbrowser
import mesa
from mesa.visualization.modules import ChartModule
from participation_model import ParticipationModel
from participation_agent import ColorCell

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
]


def participation_draw(cell: ColorCell):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the cell in its current color.

    :param cell:  the cell in the simulation

    :return: the portrayal dictionary.
    """
    if cell is None:
        raise AssertionError
    # num_agents_at_cell = cell.model.grid.get_cell_list_contents([cell.pos])
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0,
                 "x": cell.row, "y": cell.col,
                 "Color": _COLORS[cell.color]}
    return portrayal


grid_rows = 160
grid_cols = 200
cell_size = 5
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
        name="World Height", value=200, min_value=10, max_value=1000, step=10,
        description="Select the height of the world"
    ),
    "width": mesa.visualization.Slider(
        name="World Width", value=160, min_value=10, max_value=1000, step=10,
        description="Select the width of the world"
    ),
    "num_agents": mesa.visualization.Slider(
        name="# Agents", value=200, min_value=10, max_value=9999999, step=10
    ),
    "num_colors": mesa.visualization.Slider(
        name="# Colors", value=4, min_value=2, max_value=100, step=1
    ),
    # "num_regions": mesa.visualization.Slider(
    #     name="# Regions", value=4, min_value=4, max_value=500, step=1
    # ),
}
