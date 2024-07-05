from mesa.experimental import JupyterViz, make_text, Slider
from participation_model import ParticipationModel
from participation_agent import VoteAgent
from numpy import arange
import solara
from model_setup import participation_draw
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
    "height": Slider("World Height", 200, 10, 1000, 10),
    "width": Slider("World Width", 160, 10, 1000, 10),
    "num_agents": Slider("# Agents", 200, 10, 9999999, 10),
    "num_colors": Slider("# Colors", 4, 2, 100, 1),
}

page = JupyterViz(
    ParticipationModel,
    model_params,
    measures=["wealth", make_text(get_agents_assets),],
    # agent_portrayal=agent_portrayal,
    agent_portrayal=participation_draw,
    space_drawer=space_drawer,
)
page  # noqa
