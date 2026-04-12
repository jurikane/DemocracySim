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
      result[y, x] == color at position (x, y)
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
    return model.step_metrics_snapshot["collective_assets"]


def compute_gini_index(model):
    return model.step_metrics_snapshot["gini_index"]


def compute_gini_dissatisfaction(model):
    return model.step_metrics_snapshot["gini_dissatisfaction"]


def compute_group_turnout(model) -> list[float]:
    return model.step_metrics_snapshot["group_turnout"]


def compute_group_mean_assets_share(model) -> list[float]:
    return model.step_metrics_snapshot["group_mean_assets_share"]


def compute_group_mean_dissatisfaction(model) -> list[float]:
    return model.step_metrics_snapshot["group_mean_dissatisfaction"]


def get_voter_turnout(model):
    return model.step_metrics_snapshot["turnout"]


def compute_global_quality_distance(model):
    return model.step_metrics_snapshot["quality_distance"]


def compute_group_outcome_distance(model) -> list[float]:
    return model.step_metrics_snapshot["group_outcome_distance"]


def build_step_metrics_snapshot(model) -> dict[str, float | int | list[float]]:
    agents = model.voting_agents
    areas = model.areas
    num_groups = model.num_personality_groups

    collective_assets = 0.0
    altruism_sum = 0.0
    dissatisfaction_sum = 0.0
    participation_prob_sum = 0.0
    assets: list[float] = []
    dissatisfaction_values: list[float] = []
    group_assets = [[] for _ in range(num_groups)]
    group_dissatisfaction = [[] for _ in range(num_groups)]
    eligible_counts = [0] * num_groups
    participant_counts = [0] * num_groups

    for agent in agents:
        asset = agent.assets
        dissatisfaction = agent.dissatisfaction_value
        group_idx = agent.personality_group_idx

        collective_assets += asset
        altruism_sum += agent.altruism_factor
        dissatisfaction_sum += dissatisfaction
        participation_prob_sum += agent.participation_probability()
        assets.append(asset)
        dissatisfaction_values.append(dissatisfaction)
        group_assets[group_idx].append(asset)
        group_dissatisfaction[group_idx].append(dissatisfaction)

        if agent.eligible_for_election:
            eligible_counts[group_idx] += 1
            if agent.participating:
                participant_counts[group_idx] += 1

    total_agents = len(agents)
    group_mean_assets = [
        np.mean(values) if values else float("nan")
        for values in group_assets
    ]
    finite_group_mean_assets = [value for value in group_mean_assets if np.isfinite(value)]
    total_group_mean_assets = sum(finite_group_mean_assets)
    total_participants = sum(area.num_agents_participated_last or 0 for area in areas)
    total_resident = sum(area.num_agents for area in areas)
    turnout = 100.0 * total_participants / total_resident if total_resident else 0.0

    quality_weighted_sum = 0.0
    quality_total_weight = 0
    outcome_distance_weighted_sums = [0.0] * num_groups
    outcome_distance_total_weights = [0] * num_groups

    for area in areas:
        eligible_voters = area.num_eligible_voters_last
        if eligible_voters:
            quality = area.puzzle_distance if area.puzzle_mode else area.dist_to_reality
            if quality is not None and np.isfinite(quality):
                quality_weighted_sum += quality * eligible_voters
                quality_total_weight += eligible_voters

        group_distances = area.group_outcome_distance
        group_counts = area.personality_group_counts
        for group_idx, count in enumerate(group_counts):
            if not count or group_idx >= len(group_distances):
                continue
            distance = group_distances[group_idx]
            if np.isfinite(distance):
                outcome_distance_weighted_sums[group_idx] += distance * count
                outcome_distance_total_weights[group_idx] += count

    return {
        "collective_assets": collective_assets,
        "gini_index": gini_index_0_100(assets),
        "gini_dissatisfaction": gini_index_0_100(dissatisfaction_values),
        "turnout": turnout,
        "quality_distance": (
            quality_weighted_sum / quality_total_weight
            if quality_total_weight
            else 0.0
        ),
        "group_turnout": [
            100.0 * participant_counts[group_idx] / eligible_counts[group_idx]
            if eligible_counts[group_idx]
            else float("nan")
            for group_idx in range(num_groups)
        ],
        "group_mean_assets_share": [
            value / total_group_mean_assets if total_group_mean_assets > 0.0 else float("nan")
            for value in group_mean_assets
        ],
        "group_mean_dissatisfaction": [
            np.mean(values) if values else float("nan")
            for values in group_dissatisfaction
        ],
        "group_outcome_distance": [
            outcome_distance_weighted_sums[group_idx] / outcome_distance_total_weights[group_idx]
            if outcome_distance_total_weights[group_idx]
            else float("nan")
            for group_idx in range(num_groups)
        ],
        "mean_p_participation": participation_prob_sum / total_agents if total_agents else 0.0,
        "mean_altruism": altruism_sum / total_agents if total_agents else 0.0,
        "mean_dissatisfaction": dissatisfaction_sum / total_agents if total_agents else 0.0,
    }
