from __future__ import annotations
from typing import Optional
from src.utils.metrics import gini_index_0_100
from src.agents import Area
import numpy as np


def is_rate_btw_0_and_1(value, nan_allowed=False) -> float | None:
    """
    Helper function to check if a value is a valid rate in [0,1].
    Args:
    - value: The value to check.
    - nan_allowed: If True, allows None as a valid input.
    Returns:
    - The value if it's a valid rate, or None if it's None and None is allowed.
    """
    if value is None and nan_allowed:
        return None
    elif value is not None and isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
        return float(value)
    raise ValueError(f"Rate value must be in [0,1]. Got: {value} type={type(value)}")

def is_learning_rate(value) -> float:
    """
    Helper function to check if a value is a valid learning rate >= 0.
    Args:
    - value: The value to check.
    Returns:
    - The value if it's a valid learning rate.
    """
    # Learning rate. How fast q changes in response to the signal.
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Learning rates alpha must be finite and >= 0.")
    return float(value)

def get_area_voter_turnout(area: Area) -> Optional[float]:
    return area.voter_turnout if isinstance(area, Area) else None


def get_area_dist_to_reality(area: Area) -> Optional[float]:
    return area.dist_to_reality if isinstance(area, Area) else None


def get_area_puzzle_distance(area: Area) -> Optional[float]:
    return area.puzzle_distance if isinstance(area, Area) else None


def get_area_quality_distance(area: Area) -> Optional[float]:
    if not isinstance(area, Area):
        return None
    if area.puzzle_mode:
        return area.puzzle_distance
    return area.dist_to_reality


def get_area_color_distribution(area: Area) -> Optional[list[float]]:
    return area.color_distribution.tolist() if isinstance(area, Area) else None


def get_area_puzzle_distribution(area: Area) -> Optional[list[float]]:
    if not isinstance(area, Area):
        return None
    puzzle = area.puzzle_distribution
    return puzzle.tolist() if isinstance(puzzle, np.ndarray) else None


def get_election_results(area: Area) -> Optional[list[int]]:
    """
    Returns the voted ordering as a list or None if not available.

    Returns:
        list[int] | None
    """
    if isinstance(area, Area) and area.voted_ordering is not None:
        return area.voted_ordering.tolist()
    return None


def get_area_gini_index(area: Area) -> Optional[float]:
    """Per-area Gini index (0-100) computed from agents' assets.
    """
    if not isinstance(area, Area):
        return None
    assets = [a.assets for a in area.agents]
    return float(gini_index_0_100(assets))

def ensure_rate_0_1(name: str, value, *, allow_none: bool = False) -> float | None:
    """Return value as float if in [0,1]; optionally allow None."""
    if value is None and allow_none:
        return None
    if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
        return float(value)
    raise ValueError(f"{name} must be in [0,1]. Got: {value} type={type(value)}")


def ensure_choice(name: str, value, allowed: set[str]) -> str:
    """Return value if in allowed set; otherwise raise ValueError."""
    if value in allowed:
        return value
    raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}.")


def ensure_int_ge_0(name: str, value) -> int:
    """Return value as int if it is an integer >= 0 (bool is rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int >= 0. Got: {value} type={type(value)}")
    if value < 0:
        raise ValueError(f"{name} must be an int >= 0. Got: {value}")
    return int(value)


def ensure_finite_ge_0(name: str, value) -> float:
    """Return value as float if it is finite and >= 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and >= 0. Got: {value} type={type(value)}")
    return float(value)


def ensure_finite_gt_0(name: str, value) -> float:
    """Return value as float if it is finite and > 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and > 0. Got: {value} type={type(value)}")
    return float(value)
