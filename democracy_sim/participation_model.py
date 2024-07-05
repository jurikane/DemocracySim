import random
from math import sqrt
import mesa
from mesa.time import StagedActivation
from participation_agent import VoteAgent, ColorCell


class Area:  # TODO implement this
    def __init__(self, model, height, width, size_variance=0):
        if size_variance == 0:
            self._width = width
            self._height = height
            self.width_off, self.height_off = 0, 0
        elif size_variance > 1 or size_variance < 0:
            raise ValueError("Size variance must be between 0 and 1")
        else:  # Apply variance
            w_var_factor = random.uniform(1 - size_variance, 1 + size_variance)
            h_var_factor = random.uniform(1 - size_variance, 1 + size_variance)
            self._width = int(width * w_var_factor)
            self.width_off = abs(width - self._width)
            self._height = int(height * h_var_factor)
            self.height_off = abs(height - self._height)
        self.model = model
        self.agents = []
        self.cells = []
        self._idx_field = None

    @property
    def idx_field(self):
        return self._idx_field

    @idx_field.setter
    def idx_field(self, value: tuple):
        try:
            x_val, y_val = value
        except ValueError:
            raise ValueError("The idx_field must be a tuple")
        if x_val < 0 or x_val >= self.model.width:
            raise ValueError(f"The x={x_val} value must be within the grid")
        if y_val < 0 or y_val >= self.model.height:
            raise ValueError(f"The y={y_val} value must be within the grid")
        x_off = self.width_off // 2
        y_off = self.height_off // 2
        # Adjusting indices with offset and ensuring they wrap around the grid
        adjusted_x = (x_val + x_off) % self.model.width
        adjusted_y = (y_val + y_off) % self.model.height
        # Assign the cells to the area
        for x_area in range(self._width):
            for y_area in range(self._height):
                x = (adjusted_x + x_area) % self.model.width
                y = (adjusted_y + y_area) % self.model.height
                local_agents = self.model.grid.get_cell_list_contents([(x, y)])
                for a in local_agents:
                    if isinstance(a, VoteAgent):
                        self.add_agent(a)  # Add the agent to the area
                    elif isinstance(a, ColorCell):
                        a.add_area(self)  # Add the area to the cell
                        # Mark as a border cell if true
                        if (x_area == 0 or y_area == 0
                                or x_area == self._width - 1
                                or y_area == self._height - 1):
                            a.is_border_cell = True
                        self.add_cell(a)  # Add the cell to the area

        self._idx_field = (adjusted_x, adjusted_y)

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_cell(self, cell):
        self.cells.append(cell)

    def conduct_election(self, rule):
        # Placeholder for election logic
        pass


def compute_collective_assets(model):
    sum_assets = sum(agent.assets for agent in model.all_agents)
    return sum_assets


def get_num_agents(model):
    return len(model.all_agents)


class ParticipationModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, height, width, num_agents, num_colors, num_areas,
                 av_area_height, av_area_width, area_size_variance,
                 draw_borders=False):
        super().__init__()
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.all_agents = []
        self.num_colors = num_colors
        # Area variables
        self.num_areas = num_areas
        self.av_area_height = av_area_height
        self.av_area_width = av_area_width
        self.area_size_variance = area_size_variance
        # Create schedulers and assign it to the model
        self.color_cell_scheduler = mesa.time.RandomActivation(self)
        self.area_scheduler = mesa.time.RandomActivation(self)
        # self.agent_scheduler = mesa.time.RandomActivation(self)
        # self.schedule = StagedActivation(self,
        #                                  stage_list=['color_step', 'step'])
        # The grid
        # SingleGrid enforces at most one agent per cell;
        # MultiGrid allows multiple agents to be in the same cell.
        self.grid = mesa.space.MultiGrid(height=height, width=width, torus=True)
        self.draw_borders = draw_borders
        # Create color ids for the cells
        for _, (row, col) in self.grid.coord_iter():
            color = random.choice(range(num_colors))  # TODO improve this
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
            # Add the agent to the models' agents and scheduler
            self.all_agents.append(a)
            # Place at a random cell
            self.grid.place_agent(a, (x, y))
            # Count the agent at the chosen cell
            agents = self.grid.get_cell_list_contents([(x, y)])
            cell = [a for a in agents if isinstance(a, ColorCell)][0]
            cell.num_agents_in_cell = cell.num_agents_in_cell + 1
        # Create areas spread approximately evenly across the grid
        roo_apx = int(sqrt(self.num_areas))
        area_x_dist = self.grid.width // roo_apx
        area_y_dist = self.grid.height // roo_apx
        x_coords = range(0, self.grid.width, area_x_dist)
        y_coords = range(0, self.grid.height, area_y_dist)
        for x_coord in x_coords:
            for y_coord in y_coords:
                area = Area(self, av_area_height, av_area_width,
                            area_size_variance)
                print(f"Area {id(area)} at {x_coord}, {y_coord}")
                area.idx_field = (x_coord, y_coord)
                # area_height = self.dist_area_height()
                # area_width = self.dist_area_width()
                # new_area = Area(self, area_height, area_width)
                # # Add agents to the area
                # for agent in self.all_agents:
                #     if len(new_area.agents) < area_height * area_width:
                #         new_area.add_agent(agent)
                # Add the area to the model
                self.area_scheduler.add(area)
        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Collective assets": compute_collective_assets,
                "Number of agents": get_num_agents,
                # "Middle Class": get_num_mid_agents,
                # "Savings": get_total_savings,
                # "Wallets": get_total_wallets,
                # "Money": get_total_money,
                # "Loans": get_total_loans,
            },
            agent_reporters={"Wealth": lambda x: getattr(x, "assets", None)},
        )
        # Collect initial data
        self.datacollector.collect(self)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now
        # this will call the step method of each agent
        # and print the agent's unique_id
        self.color_cell_scheduler.step()
        self.datacollector.collect(self)  # Collect data at each step

    # def dist_area_height(self):
    #     av_height = self.av_area_height
    #     variance = self.area_size_variance
    #     return self.random.randint(
    #             av_height - variance * av_height,
    #             av_height + variance * av_height
    #         )
    #
    # def dist_area_width(self):
    #     av_width = self.av_area_width
    #     variance = self.area_size_variance
    #     return self.random.randint(
    #             av_width - variance * av_width,
    #             av_width + variance * av_width
    #         )