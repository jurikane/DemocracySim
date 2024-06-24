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
        self.color_cell_scheduler = mesa.time.RandomActivation(self)
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
        # Create color ids for the cells
        for _, (row, col) in self.grid.coord_iter():
            color = random.choice(range(num_colors))
            cell = ColorCell((row, col), self, color)
            self.grid.place_agent(cell, (row, col))
            # Add the cell color to the scheduler
            self.color_cell_scheduler.add(cell)
        # Create agents
        for a_id in range(self.num_agents):
            # Get a random position
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            a = VoteAgent(a_id, (x, y), self)
            # Add the agent to the scheduler
            self.agent_scheduler.add(a)
            # Place at a random cell
            self.grid.place_agent(a, (x, y))
            # Count the agent at the chosen cell
            agents = self.grid.get_cell_list_contents([(x, y)])
            cell = [a for a in agents if isinstance(a, ColorCell)][0]
            cell.num_agents_in_cell = cell.num_agents_in_cell + 1

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now
        # this will call the step method of each agent
        # and print the agent's unique_id
        self.color_cell_scheduler.step()
        self.agent_scheduler.step()
