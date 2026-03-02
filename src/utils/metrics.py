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


def compute_collective_assets(model):
    sum_assets = sum(agent.assets for agent in model.voting_agents)
    return sum_assets


def compute_gini_index(model):
    assets = [agent.assets for agent in model.voting_agents]
    return gini_index_0_100(assets)


def get_voter_turnout(model):
    """Return global voter turnout in percent across area-election resident population.

    Turnout semantics:
      100 * (sum participants across stepped areas) / (sum resident agents across stepped areas)
    """
    total_participants = float(sum(int(area.num_agents_participated_last or 0) for area in model.areas))
    total_resident = float(sum(int(area.num_agents) for area in model.areas))
    return (100.0 * total_participants / total_resident) if total_resident > 0.0 else 0.0

