from __future__ import annotations

import types

import numpy as np
import pytest

import src.models.participation_model as pm_mod
from src.utils.ballots import score_options_c2
from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _single_agent_model(seed: int = 9401):
    model, _ = create_test_model(
        seed=seed,
        num_agents=1,
        num_areas=1,
        num_colors=4,
        num_personality_groups=4,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=8,
        max_steps=1,
    )
    return model, model.areas[0], model.voting_agents[0]


def test_altruistic_score_row_cache_matches_direct_and_reuses_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _area = create_test_model(
        seed=9402,
        num_agents=6,
        num_areas=1,
        num_colors=4,
        num_personality_groups=4,
        known_cells=8,
        max_steps=1,
    )
    ordering = np.asarray(model.options[7], dtype=np.int64)

    calls = {"n": 0}

    def _wrapped_score_options_c2(*, target_ordering, options, distance_func, color_search_pairs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return score_options_c2(
            target_ordering=target_ordering,
            options=options,
            distance_func=distance_func,
            color_search_pairs=color_search_pairs,
        )

    monkeypatch.setattr(pm_mod, "score_options_c2", _wrapped_score_options_c2)

    s1 = model.get_altruistic_oppose_scores_for_ordering(ordering)
    s2 = model.get_altruistic_oppose_scores_for_ordering(ordering)

    direct = score_options_c2(
        target_ordering=ordering,
        options=np.asarray(model.options),
        distance_func=model.distance_func,
        color_search_pairs=model.color_search_pairs,
    )
    assert np.allclose(s1, direct, atol=1e-12)
    assert np.allclose(s2, direct, atol=1e-12)
    assert calls["n"] == 1
    assert model.altruistic_score_cache_hits >= 1
    assert model.altruistic_score_cache_misses == 1
    assert s1 is s2
    assert s1.flags.writeable is False
    with pytest.raises(ValueError):
        s1[0] = np.float32(0.0)


def test_altruistic_score_row_cache_raises_on_invalid_ordering() -> None:
    model, _ = create_test_model(
        seed=9403,
        num_agents=6,
        num_areas=1,
        num_colors=4,
        num_personality_groups=4,
        known_cells=8,
        max_steps=1,
    )
    with pytest.raises(ValueError, match="ordering"):
        model.get_altruistic_oppose_scores_for_ordering(np.asarray([99, 1, 2, 3], dtype=np.int64))


def test_default_voting_strategy_altruistic_path_uses_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    model, area, agent = _single_agent_model(seed=9404)
    agent.altruism_factor = 1.0

    target_ordering = np.asarray(model.options[11], dtype=np.int64)
    dist = np.zeros(model.num_colors, dtype=np.float32)
    for rank, color in enumerate(target_ordering):
        dist[int(color)] = np.float32(model.num_colors - rank)
    dist /= float(np.sum(dist))

    def _fake_est(self, _area):  # type: ignore[no-untyped-def]
        self.est_real_dist = np.asarray(dist, dtype=np.float32)
        self.confidence = 1.0
        return self.est_real_dist, self.confidence

    agent.estimate_real_distribution = types.MethodType(_fake_est, agent)

    calls = {"n": 0}
    original = model.get_altruistic_oppose_scores_for_ordering

    def _wrapped(ordering):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(ordering)

    monkeypatch.setattr(model, "get_altruistic_oppose_scores_for_ordering", _wrapped)
    agent.update_known_cells(area)
    scores = np.asarray(agent.vote(area), dtype=np.float32)

    assert agent.voted_altruistically is True
    assert calls["n"] == 1
    assert scores.shape[0] == np.asarray(model.options).shape[0]

