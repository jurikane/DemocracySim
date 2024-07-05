from mesa.experimental import JupyterViz, make_text, Slider
import solara
from model_setup import *
# Data visualization tools.
from matplotlib.figure import Figure


def get_agents_assets(model: ParticipationModel):
    """
    Display a text count of how many happy agents there are.
    """
    all_assets = list()
    # Store the results
    for agent in model.all_agents:
        all_assets.append(agent.assets)
    return f"Agents wealth: {all_assets}"


def agent_portrayal(agent: VoteAgent):
    # Construct and return the portrayal dictionary
    portrayal = {
        "size": agent.assets,
        "color": "tab:orange",
    }
    return portrayal


def space_drawer(model, agent_portrayal):
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.subplots()

    # Set plot limits and aspect
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Match grid's origin

    fig.tight_layout()

    return solara.FigureMatplotlib(fig)


model_params = {
    "height": grid_rows,
    "width": grid_cols,
    "num_agents": Slider("# Agents", 200, 10, 9999999, 10),
    "num_colors": Slider("# Colors", 4, 2, 100, 1),
    "num_areas": Slider("# Areas", num_areas, 4, min(grid_cols, grid_rows)//2, 1),
    "av_area_height": Slider("Av. Area Height", area_height, 2, grid_rows//2, 1),
    "av_area_width": Slider("Av. Area Width", area_width, 2, grid_cols//2, 1),
    "area_size_variance": Slider("Area Size Variance", area_var, 0.0, 1.0, 0.1),
    "draw_borders": False,
}

page = JupyterViz(
    ParticipationModel,
    model_params,
    #measures=["wealth", make_text(get_agents_assets),],
    # agent_portrayal=agent_portrayal,
    agent_portrayal=participation_draw,
    space_drawer=space_drawer,
)
page  # noqa
