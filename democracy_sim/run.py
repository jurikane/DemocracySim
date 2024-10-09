import mesa
from democracy_sim.participation_model import ParticipationModel
from democracy_sim.model_setup import (model_params, canvas_element,
                                       voter_turnout, wealth_chart,
                                       color_distribution_chart)
from democracy_sim.visualisation_elements import *


color_distribution_element = ColorDistributionElement()
steps_text = StepsTextElement()
vto_areas = VoterTurnoutElement()


server = mesa.visualization.ModularServer(
    model_cls=ParticipationModel,
    visualization_elements=[canvas_element, color_distribution_chart,
                            wealth_chart, voter_turnout, vto_areas,
                            color_distribution_element, steps_text],
    name="DemocracySim",
    model_params=model_params,
)

server.launch(open_browser=True)
