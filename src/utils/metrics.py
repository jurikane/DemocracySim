import numpy as np
from typing import Sequence, Union


def gini_index_0_100(values: Union[Sequence[float], np.ndarray, None]) -> int:
    """Compute the Gini index (0-100) for a 1D sequence of non-negative values.

    Edge cases:
      - empty / 1 element -> 0
      - all zeros -> 0
    """
    if values is None:
        return 0
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = int(arr.size)
    if n <= 1:
        return 0
    total = float(arr.sum())
    if total <= 0:
        return 0

    # Sort ascending (O(n log n)); n per-area is typically small.
    s = np.sort(arr)
    # Standard gini formula: (2*sum(i*x_i))/(n*sum(x)) - (n+1)/n
    i = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float((i * s).sum())) / (n * total) - (n + 1.0) / n
    # Numerical safety
    if g < 0:
        g = 0.0
    if g > 1:
        g = 1.0
    return int(g * 100)


def get_grid_colors(model):
    """
    Return the current grid state as an array of rows (row-major):
      result[y][x] == color at position (x, y)
    """
    grid = model.grid

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
        (cell.is_border_cell for cell, _pos in grid.coord_iter()),
        dtype=bool,
        count=w * h,
    )
    return flat.reshape((w, h)).T  # -> shape (h, w) with arr[y, x]


def compute_collective_assets(model):
    sum_assets = sum(agent.assets for agent in model.voting_agents)
    return sum_assets


def compute_gini_index(model):
    assets = [agent.assets for agent in model.voting_agents]
    return gini_index_0_100(assets)


def get_voter_turnout(model):
    """Return global voter turnout as the mean across *stepped* areas.

    Note: the model currently does NOT run elections for `model.global_area`.
    If global elections are ever implemented, they should be included here
    explicitly (and only then).
    """
    voter_turnout_sum = 0.0
    num_areas = int(model.num_areas)
    if num_areas == 0:
        return 0
    for area in model.areas:
        voter_turnout_sum += float(area.voter_turnout)
    return voter_turnout_sum / num_areas


def get_agents_per_cell_grid(model) -> np.ndarray:
    """Return an HxW int grid with the number of voting agents per cell.

    Contract:
      result[y][x] == number of agents in the ColorCell at position (x, y)

    Implementation mirrors get_grid_colors/get_area_border_grid ordering.
    """
    grid = model.grid
    h, w = grid.height, grid.width
    flat = np.fromiter(
        (len(cell.agents) if cell is not None else 0 for cell, _pos in grid.coord_iter()),
        dtype=np.int32,
        count=w * h,
    )
    return flat.reshape((w, h)).T  # -> shape (h, w) with arr[y, x]


def get_agent_strings_per_cell_grid(model) -> np.ndarray:
    """Return an HxW str grid with vote agent infos per cell.

    Contract:
      result[y][x] == str listing all vote agents in the ColorCell at (x, y)
    """
    grid = model.grid
    h, w = grid.height, grid.width

    def agents_to_str(agents) -> str:
        return ", ".join(f"{a.unique_id}: {a.personality_group}" for a in agents)

    flat = [
        agents_to_str(cell.agents) if cell is not None else ""
        for cell, _pos in grid.coord_iter()
    ]
    return np.asarray(flat, dtype=object).reshape((w, h)).T


def get_area_strings_per_cell_grid(model) -> np.ndarray:
    """Return an HxW str grid with area ids per cell.
    Contract:
      result[y][x] == str listing all area ids in the ColorCell at (x, y)
      id-sting: "a1, a2, ..." (excluding global area with id -1)
    """
    grid = model.grid
    h, w = grid.height, grid.width

    def areas_to_str(areas) -> str:
        return ", ".join(f"{a.unique_id}" for a in areas if a.unique_id != -1)
    flat = [
        areas_to_str(cell.areas) if cell is not None else ""
        for cell, _pos in grid.coord_iter()
    ]
    return np.asarray(flat, dtype=object).reshape((w, h)).T
