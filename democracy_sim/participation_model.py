from typing import TYPE_CHECKING, cast
import random
from math import sqrt
import mesa
from mesa import Agent
from participation_agent import VoteAgent, ColorCell
import numpy as np
from itertools import permutations, product, combinations

election_cost = 5  # TODO: integrate properly


class Area(Agent):
    def __init__(self, unique_id, model, height, width, size_variance):
        if TYPE_CHECKING:  # Type hint for IDEs
            model = cast(ParticipationModel, model)
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
        self._idx_field = None  # An indexing position of the area in the grid
        self.color_distribution = [0] * model.num_colors  # Initialize to 0
        self.voted_distribution = [0] * model.num_colors

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
        self.update_color_distribution()
        self._idx_field = (adjusted_x, adjusted_y)

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_cell(self, cell):
        self.cells.append(cell)

    def conduct_election(self, voting_rule, distance_func):
        """
        This method holds the primary logic of the simulation by simulating
        the election in the area as well as handling the payments and rewards.
        """
        # Ask agents to participate
        participating_agents = []
        preference_profile = []
        for agent in self.agents:
            if agent.ask_for_participation(area=self):
                participating_agents.append(agent)
                # collect the participation fee from the agents
                agent.assets = agent.assets - election_cost
                # Ask participating agents for their prefs
                preference_profile.append(agent.vote(area=self))  # TODO use np!
        # TODO: WHERE to discretize if needed?
        # accumulate the prefs using the voting rule
        aggreg_prefs = voting_rule(preference_profile)
        # save the "elected" distribution in self.voted_distribution
        winning_option = aggreg_prefs[0]
        self.voted_distribution = self.model.options[winning_option]
        # calculate the distance to the real distribution using distance_func
        distance_factor = distance_func(self.voted_distribution,
                                        self.color_distribution)
        # calculate the rewards for the agents
        # TODO
        # distribute the rewards
        # TODO

    def update_color_distribution(self):
        """
        This method calculates the current color distribution of the area
        and saves it in the color_distribution attribute.
        """
        color_count = {}
        num_cells = len(self.cells)
        for cell in self.cells:
            color = cell.color
            color_count[color] = color_count.get(color, 0) + 1
        for color in range(self.model.num_colors):
            dist_val = color_count.get(color, 0) / num_cells  # Float division
            self.color_distribution[color] = dist_val
        print(f"Area {self.unique_id} color "
              f"distribution: {self.color_distribution}")

    def step(self) -> None:
        self.update_color_distribution()
        self.conduct_election(self.model.voting_rule)


def compute_collective_assets(model):
    sum_assets = sum(agent.assets for agent in model.all_agents)
    return sum_assets


def get_num_agents(model):
    return len(model.all_agents)


def get_area_color_distributions(model):
    return {area.unique_id: area.color_distribution
            for area in model.area_scheduler.agents}


def color_by_dst(color_distribution) -> int:
    """
    This method selects a color (int) of range(len(color_distribution))
    such that, each color is selected with a probability according to the
    given color_distribution array.
    Example: color_distribution = [0.2, 0.3, 0.5]
    Color 1 is selected with a probability of 0.3
    """
    r = random.random()
    for color_idx, prob in enumerate(color_distribution):
        if r < prob:
            return color_idx
        r -= prob


def create_all_options(n, include_ties=False):
    """
    Creates and returns the list of all possible ranking vectors,
    if specified including ties.
    Rank values start from 0.
    :param n: The number of items to rank (number of colors in our case)
    :param include_ties: If True, rankings include ties.
    :return r: A NumPy matrix containing all possible rankings of n items
    """
    if include_ties:
        # Create all possible combinations and sort out invalid rankings
        # i.e. [1, 1, 1] or [1, 2, 2] aren't valid as no option is ranked first.
        r = np.array([np.array(comb) for comb in product(range(n), repeat=n)
                      if set(range(max(comb))).issubset(comb)])
    else:
        r = np.array([np.array(p) for p in permutations(range(n))])
    return r


class ParticipationModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, height, width, num_agents, num_colors, num_areas,
                 av_area_height, av_area_width, area_size_variance, patch_power,
                 color_patches_steps, draw_borders, heterogeneity, voting_rule):
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
        self.heterogeneity = heterogeneity
        # Random bias factors that affect the initial color distribution
        self.vertical_bias = random.uniform(0, 1)
        self.horizontal_bias = random.uniform(0, 1)
        self.draw_borders = draw_borders
        # Color distribution
        self.color_dst = self.create_color_distribution(heterogeneity)
        # Elections
        self.voting_rule = voting_rule
        self.options = create_all_options(num_colors)
        # Create search pairs once for faster iterations when comparing rankings
        self.search_pairs = combinations(range(0, num_colors), 2)
        self.color_vec = np.arange(num_colors)  # Also for faster algorithms
        # Create color ids for the cells
        for _, (row, col) in self.grid.coord_iter():
            color = color_by_dst(self.color_dst)
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
                "Area Color Distributions": get_area_color_distributions,
            },
            agent_reporters={"Wealth": lambda ag: getattr(ag, "assets", None)},
        )
        # Adjust the color pattern to make it less random (see color patches)
        for _ in range(color_patches_steps):
            print(f"Color adjustment step {_}")
            for cell in self.grid.coord_iter():
                agents = cell[0]
                if TYPE_CHECKING:
                    agents = cast(list, agents)
                c = [cell for cell in agents if isinstance(cell, ColorCell)][0]
                most_common_color = self.color_patches(c, patch_power)
                c.color = most_common_color
        # Collect initial data
        self.datacollector.collect(self)

    def step(self):
        """Advance the model by one step."""

        # Conduct elections in the areas
        self.area_scheduler.step()
        # Mutate the color cells according to election outcomes
        self.color_cell_scheduler.step()
        # Collect data for monitoring and data analysis
        self.datacollector.collect(self)

    def create_color_distribution(self, heterogeneity):
        """
        This method is used to create a color distribution that has a bias
        according to the given heterogeneity factor.
        """
        colors = range(self.num_colors)
        values = [abs(random.gauss(1, heterogeneity)) for _ in colors]
        # Normalize (with float division)
        total = sum(values)
        dst_array = [value / total for value in values]
        print(f"Color distribution: {dst_array}")
        return dst_array

    def color_patches(self, cell, patch_power):
        """
        This method is used to create a less random initial color distribution
        using a similar logic to the color patches model.
        It uses a (normalized) bias coordinate to center the impact of the
        color patches structures impact around.
        :param cell: The cell that may change its color accordingly
        :param patch_power: Like a radius of impact around the bias point.
        """
        # Calculate the normalized position of the cell
        normalized_x = cell.row / self.height
        normalized_y = cell.col / self.width
        # Calculate the distance of the cell to the bias point
        bias_factor = (abs(normalized_x - self.horizontal_bias)
                       + abs(normalized_y - self.vertical_bias))
        # The closer the cell to the bias-point, the less often it is
        # to be replaced by a color chosen from the initial distribution:
        if abs(random.gauss(0, patch_power)) < bias_factor:
            return color_by_dst(self.color_dst)
        # Otherwise, apply the color patches logic
        neighbor_cells = self.grid.get_neighbors((cell.row, cell.col),
                                                 moore=True,
                                                 include_center=False)
        color_counts = {}  # Count neighbors' colors
        for neighbor in neighbor_cells:
            if isinstance(neighbor, ColorCell):
                color = neighbor.color
                color_counts[color] = color_counts.get(color, 0) + 1
        if color_counts:
            max_count = max(color_counts.values())
            most_common_colors = [color for color, count in color_counts.items()
                                  if count == max_count]
            return self.random.choice(most_common_colors)
        return cell.color  # Return the cell's own color if no consensus
