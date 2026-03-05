from __future__ import annotations

from src.config.schema import ModelConfig
from src.model_setup import build_model_params


def test_build_model_params_exposes_seed_control() -> None:
    """Guardrail: seed must be controllable from the live Mesa UI params."""

    cfg = ModelConfig(
        height=10,
        width=10,
        num_agents=10,
        initial_agent_assets=100.0,
        num_colors=3,
        color_patches_steps=0,
        patch_power=1.0,
        heterogeneity=0.1,
        known_cells=3,
        num_personality_groups=2,
        num_areas=1,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
        election_cost_rate=0.1,
        election_impact_on_mutation=1.0,
        mu=0.1,
        rule_idx=0,
        distance_idx=0,
        participation_alpha=0.05,
        participation_beta=1.0,
        participation_init_q=0.0,
        participation_q_max=50.0,
        altruism_alpha=0.05,
        altruism_init=0.5,
        altruism_clip_min=0.0,
        altruism_clip_max=1.0,
        personal_preference_peakedness=1.0,
        seed=123,
    )

    params = build_model_params(cfg)
    assert "seed" in params
