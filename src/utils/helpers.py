from __future__ import annotations
from typing import Optional
from src.utils.metrics import gini_index_0_100
from src.agents import Area


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


def get_area_voter_turnout(area: Area) -> Optional[float]:
    return area.voter_turnout if isinstance(area, Area) else None


def get_area_dist_to_reality(area: Area) -> Optional[float]:
    return area.dist_to_reality if isinstance(area, Area) else None


def get_area_color_distribution(area: Area) -> Optional[list[float]]:
    return area.color_distribution.tolist() if isinstance(area, Area) else None


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
