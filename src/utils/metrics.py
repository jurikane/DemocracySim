import numpy as np


def get_grid_colors(model):
    """
    Return the current grid state as an array of rows (row-major):
      result[y][x] == color at position (x, y)
    """
    grid = getattr(model, "grid", None)
    if grid is None:
        return []

    h, w = grid.height, grid.width
    # Read in Mesa coord_iter() order (x-major), then reshape and transpose to (h, w)
    flat = np.fromiter(
        (cell.color if cell is not None else None for cell, _pos in grid.coord_iter()),
        dtype=np.int64,
        count=w * h,
    )
    grid_colors_arr = flat.reshape((w, h)).T  # -> shape (h, w) with arr[y, x]
    return grid_colors_arr


def get_area_border_grid(model):
    """
    Return the area borders grid state as an array of rows (row-major):
      result[y][x] == is_border_cell at position (x, y)
    """
    grid = model.grid
    h, w = grid.height, grid.width
    flat = np.fromiter(
        (getattr(cell, "is_border_cell", False) for cell, _pos in grid.coord_iter()),
        dtype=bool,
        count=w * h,
    )
    return flat.reshape((w, h)).T  # -> shape (h, w) with arr[y, x]


def compute_collective_assets(model):
    sum_assets = sum(agent.assets for agent in model.voting_agents)
    return sum_assets


def compute_gini_index(model):
    # TODO: separate to be able to calculate it zone-wise as well as globally
    # TODO: Unit-test this function
    # Extract the list of assets for all agents
    assets = [agent.assets for agent in model.voting_agents]
    n = len(assets)
    if n == 0:
        return 0  # No agents, no inequality
    # Sort the assets
    sorted_assets = sorted(assets)
    # Calculate the Gini Index
    cumulative_sum = sum((i + 1) * sorted_assets[i] for i in range(n))
    total_sum = sum(sorted_assets)
    if total_sum == 0:
        return 0  # No agent has any assets => view as total equality
    gini_index = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
    return int(gini_index * 100)  # Return in "percent" (0-100)


def get_voter_turnout(model):
    voter_turnout_sum = 0
    num_areas = model.num_areas
    for area in model.areas:
        voter_turnout_sum += area.voter_turnout
    if not model.global_area is None:
        # TODO: Check the correctness and whether it makes sense to include the global area here
        voter_turnout_sum += model.global_area.voter_turnout
        num_areas += 1
    elif num_areas == 0:
        return 0
    return voter_turnout_sum / num_areas
