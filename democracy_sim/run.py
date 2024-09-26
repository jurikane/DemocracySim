# import webbrowser
import mesa
from democracy_sim.participation_model import ParticipationModel
from democracy_sim.model_setup import (model_params, canvas_element,
                                       voter_turnout, wealth_chart,
                                       color_distribution_chart)

server = mesa.visualization.ModularServer(
    model_cls=ParticipationModel,
    visualization_elements=[canvas_element, color_distribution_chart,
                            wealth_chart, voter_turnout],
    name="DemocracySim",
    model_params=model_params,
)

server.launch(open_browser=True)
