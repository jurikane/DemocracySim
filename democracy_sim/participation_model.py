import random

import mesa
from mesa.time import StagedActivation
from participation_agent import VoteAgent, ColorCell


class Area:
    def __init__(self, model, width, height):
        self.model = model
        self.width = width
        self.height = height
        self.schedule = mesa.time.RandomActivation(self.model)
        self.agents = []

    def add_agent(self, agent):
        self.schedule.add(agent)
        self.agents.append(agent)

    def step(self):
        self.schedule.step()


class ParticipationModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, num_agents, num_colors, width, height):
        super().__init__()
        self.num_agents = num_agents
        self.num_colors = num_colors
        self.height = height
        self.width = width
        # Create schedulers and assign it to the model
        self.color_scheduler = mesa.time.RandomActivation(self)
        self.agent_scheduler = mesa.time.RandomActivation(self)
        # self.schedule = StagedActivation(self,
        #                                  stage_list=['color_step', 'step'])
        # The grid
        # SingleGrid enforces at most one agent per cell;
        # MultiGrid allows multiple agents to be in the same cell.
        self.grid = mesa.space.MultiGrid(width, height, torus=True)
        self.datacollector = mesa.DataCollector(
            model_reporters={"wealth": "wealth"},
            # Model-level count of agents' wealth
        )
        # Create colors for the cells
        for _, (row, col) in self.grid.coord_iter():
            color = random.choice(range(num_colors))
            cell = ColorCell((row, col), self, color, num_colors)
            self.grid.place_agent(cell, (row, col))
            # Add the cell color to the scheduler
            self.color_scheduler.add(cell)
        # Create agents
        for a_id in range(self.num_agents):
            a = VoteAgent(a_id, self)
            # Add the agent to the scheduler
            self.agent_scheduler.add(a)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now
        # this will call the step method of each agent
        # and print the agent's unique_id
        self.color_scheduler.step()
        self.agent_scheduler.step()
