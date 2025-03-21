from democracy_sim.participation_model import ParticipationModel


def create_default_model(**overrides):
    """Create a ParticipationModel instance, with optional parameter overrides."""
    params = {
        "height": 100,
        "width": 80,
        "num_agents": 800,
        "num_colors": 3,
        "num_personalities": 4,
        "mu": 0.05,
        "election_impact_on_mutation": 1.8,
        "common_assets": 40000,
        "known_cells": 10,
        "num_areas": 16,
        "av_area_height": 25,
        "av_area_width": 20,
        "area_size_variance": 0.0,
        "patch_power": 1.0,
        "color_patches_steps": 3,
        "draw_borders": True,
        "heterogeneity": 0.3,
        "rule_idx": 1,
        "distance_idx": 1,
        "election_costs": 1,
        "max_reward": 50,
        "show_area_stats": False
    }
    params.update(overrides)
    return ParticipationModel(**params)
