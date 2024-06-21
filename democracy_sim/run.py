# import webbrowser
import mesa
from participation_model import ParticipationModel
from model_setup import model_params, canvas_element, happy_chart, wealth_chart

server = mesa.visualization.ModularServer(
    model_cls=ParticipationModel,
    visualization_elements=[canvas_element, wealth_chart, happy_chart],
    name="DemocracySim",
    model_params=model_params,
)

server.launch(open_browser=True)
