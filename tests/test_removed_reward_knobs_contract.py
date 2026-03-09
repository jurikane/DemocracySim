from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from src.analysis import doe_runner
from src.config.loader import load_config
from src.config.schema import ModelConfig
from src.models.participation_model import ParticipationModel

REMOVED_KNOBS = {
    "reward_rate_common",
    "break_even_distance_personal",
    "abstention_share",
}


def test_model_config_rejects_removed_reward_knobs() -> None:
    cfg = load_config("toy.yaml")
    base = cfg.model.model_dump()
    for key in REMOVED_KNOBS:
        payload = dict(base)
        payload[key] = 0.123
        with pytest.raises(ValidationError):
            ModelConfig.model_validate(payload)


def test_participation_model_signature_excludes_removed_knobs() -> None:
    params = set(inspect.signature(ParticipationModel.__init__).parameters.keys())
    for key in REMOVED_KNOBS:
        assert key not in params


def test_doe_ranges_and_constraints_exclude_removed_knobs() -> None:
    for key in REMOVED_KNOBS:
        assert key not in doe_runner.DEFAULT_DOE_RANGES
        assert key not in doe_runner.DEFAULT_FROZEN_MODEL

    sampled = {
        "election_cost_rate": 0.1,
        "reward_rate_personal": 0.2,
        "break_even_distance_common": 0.3,
        "election_impact_on_mutation": 1.2,
        "mu": 0.5,
        "participation_alpha": 0.1,
        "participation_beta": 2.0,
        "participation_init_q": 0.2,
        "altruism_static": 0.4,
    }
    assert doe_runner._passes_rate_sum_constraint(sampled)
