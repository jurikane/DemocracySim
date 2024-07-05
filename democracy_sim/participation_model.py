from typing import TYPE_CHECKING, cast
import random
from math import sqrt
import mesa
from mesa import Agent
from participation_agent import VoteAgent, ColorCell


class Area(Agent):
    def __init__(self, unique_id, model, height, width, size_variance=0):
        super().__init__(unique_id, model)
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
        self.agents = []
        self.cells = []
        self._idx_field = None

    @property
    def idx_field(self):
        return self._idx_field

    @idx_field.setter
    def idx_field(self, value: tuple):
        if TYPE_CHECKING:  # Type hint for IDEs
            self.model = cast(ParticipationModel, self.model)
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
                 color_adj_steps, draw_borders, heterogeneity):
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
        self.color_adj_steps = color_adj_steps
        self.heterogeneity = heterogeneity
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
        roo_apx = round(sqrt(self.num_areas))
        nr_areas_x = self.grid.width // av_area_width
        nr_areas_y = self.grid.width // av_area_height
        area_x_dist = self.grid.width // roo_apx
        area_y_dist = self.grid.height // roo_apx
        print(f"roo_apx: {roo_apx}, nr_areas_x: {nr_areas_x}, "
              f"nr_areas_y: {nr_areas_y}, area_x_dist: {area_x_dist}, "
              f"area_y_dist: {area_y_dist}")
        # if (abs(nr_areas_x * nr_areas_y - self.num_areas) <
        #         abs(roo_apx**2 - self.num_areas)):
        #     area_x_dist = self.grid.width // nr_areas_x
        #     area_y_dist = self.grid.height // nr_areas_y
        #     print(f"## {nr_areas_x * nr_areas_y} vs {roo_apx**2}")
        # x_coords = [(0 + i * area_x_dist) % width for i in range(nr_areas_x)]
        # y_coords = [(0 + i * area_y_dist) % height for i in range(nr_areas_y)]
        x_coords = range(0, self.grid.width, area_x_dist)
        y_coords = range(0, self.grid.height, area_y_dist)
        # Add additional areas if necessary (num_areas not a square number)
        additional_x, additional_y = [], []
        missing = self.num_areas - len(x_coords) * len(y_coords)
        for _ in range(missing):
            additional_x.append(self.random.randrange(self.grid.width))
            additional_y.append(self.random.randrange(self.grid.height))
        a_ids = iter(range(1, self.num_areas + 1))
        for x_coord in x_coords:
            for y_coord in y_coords:
                a_id = next(a_ids, 0)
                if a_id == 0:
                    break
                area = Area(a_id, self, av_area_height, av_area_width,
                            area_size_variance)
                print(f"Area {area.unique_id} at {x_coord}, {y_coord}")
                area.idx_field = (x_coord, y_coord)
                self.area_scheduler.add(area)
        for x_coord, y_coord in zip(additional_x, additional_y):
            area = Area(next(a_ids), self, av_area_height, av_area_width,
                        area_size_variance)
            print(f"++ Area {area.unique_id} at {x_coord}, {y_coord}")
            area.idx_field = (x_coord, y_coord)
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
            agent_reporters={"Wealth": lambda ag: getattr(ag, "assets", None)},
        )
        # Adjust the color pattern to make it less random (see color patches)
        for _ in range(color_adj_steps):
            print(f"Color adjustment step {_}")
            for cell in self.grid.coord_iter():
                agents = cell[0]
                if TYPE_CHECKING:
                    agents = cast(list, agents)
                c = [cell for cell in agents if isinstance(cell, ColorCell)][0]
                if isinstance(c, ColorCell):
                    most_common_color = self.mix_colors(c, self.heterogeneity)
                    c.color = most_common_color
        # Collect initial data
        self.datacollector.collect(self)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now
        # this will call the step method of each agent
        # and print the agent's unique_id
        self.color_cell_scheduler.step()
        self.datacollector.collect(self)  # Collect data at each step

    def mix_colors(self, cell, heterogeneity):
        """
        This method is used to create a less random initial color distribution
        """
        neighbor_cells = self.grid.get_neighbors((cell.row, cell.col),
                                                 moore=True,
                                                 include_center=False)
        color_counts = {}
        for neighbor in neighbor_cells:
            if random.random() < heterogeneity:  # Create heterogeneity
                # Introduce a bias based on coordinates
                p = (cell.row * cell.col) / (self.grid.width * self.grid.height)
                if random.random() < p:
                    return random.choice(range(self.num_colors))
            if isinstance(neighbor, ColorCell):
                color = neighbor.color
                color_counts[color] = color_counts.get(color, 0) + 1
        if color_counts:
            max_count = max(color_counts.values())
            most_common_colors = [color for color, count in color_counts.items()
                                  if count == max_count]
            # if random.random() < randomness:
            #     return random.choice(range(cell.model.num_colors))  # Add some randomness
            return self.random.choice(most_common_colors)
        return cell.color  # Return the cell's own color no consensus
