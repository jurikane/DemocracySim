from __future__ import annotations

import json

import numpy as np
import pytest

from src.logging.run_logger import RunLogger
from tests.factory import create_test_model


def _agent_dists(model) -> np.ndarray:
    d = [np.asarray(a.personal_opt_dist, dtype=np.float64) for a in model.voting_agents if a is not None]
    return np.vstack(d)


def _mean_entropy(dists: np.ndarray) -> float:
    eps = 1e-12
    x = np.clip(dists, eps, 1.0)
    return float(np.mean(-np.sum(x * np.log(x), axis=1)))


def test_personal_preference_peakedness_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(personal_preference_peakedness=0.0)
    with pytest.raises(ValueError):
        create_test_model(personal_preference_peakedness=-0.5)
    with pytest.raises(ValueError):
        create_test_model(personal_preference_peakedness=float("nan"))
    with pytest.raises(ValueError):
        create_test_model(personal_preference_peakedness=float("inf"))
    with pytest.raises(ValueError):
        create_test_model(personal_preference_peakedness=True)


def test_personal_preference_peakedness_oracle_distributions_valid():
    model, _ = create_test_model(
        seed=1601,
        num_agents=20,
        num_colors=4,
        personal_preference_peakedness=1.0,
    )
    d = _agent_dists(model)
    assert d.shape == (20, 4)
    assert np.isfinite(d).all()
    assert np.all(d >= 0.0)
    np.testing.assert_allclose(np.sum(d, axis=1), np.ones(20), rtol=0.0, atol=1e-6)


def test_personal_preference_peakedness_metamorphic_higher_means_more_peaked():
    base = dict(seed=1602, num_agents=50, num_colors=4, num_personality_groups=4)
    m_flat, _ = create_test_model(**base, personal_preference_peakedness=0.5)
    m_peak, _ = create_test_model(**base, personal_preference_peakedness=2.0)

    d_flat = _agent_dists(m_flat)
    d_peak = _agent_dists(m_peak)

    mean_max_flat = float(np.mean(np.max(d_flat, axis=1)))
    mean_max_peak = float(np.mean(np.max(d_peak, axis=1)))
    assert mean_max_peak > mean_max_flat

    ent_flat = _mean_entropy(d_flat)
    ent_peak = _mean_entropy(d_peak)
    assert ent_peak < ent_flat


def test_personal_preference_peakedness_integration_static_json_reflects_change(tmp_path):
    base = dict(seed=1603, num_agents=20, num_colors=4, num_personality_groups=4)
    m_flat, _ = create_test_model(**base, personal_preference_peakedness=0.5)
    m_peak, _ = create_test_model(**base, personal_preference_peakedness=2.0)

    l1 = RunLogger(out_dir=tmp_path / "flat", run_seed=1, rule_idx=int(m_flat.rule_idx), num_steps=1, store_grid=False)
    l2 = RunLogger(out_dir=tmp_path / "peak", run_seed=1, rule_idx=int(m_peak.rule_idx), num_steps=1, store_grid=False)
    l1.write_static(m_flat)
    l2.write_static(m_peak)

    s1 = json.loads((tmp_path / "flat" / "static.json").read_text(encoding="utf-8"))
    s2 = json.loads((tmp_path / "peak" / "static.json").read_text(encoding="utf-8"))
    p1 = np.asarray(list(s1["personal_opt_dist"].values()), dtype=np.float64)
    p2 = np.asarray(list(s2["personal_opt_dist"].values()), dtype=np.float64)

    # Integration expectation: the logged static distributions reflect the knob effect.
    assert float(np.mean(np.max(p2, axis=1))) > float(np.mean(np.max(p1, axis=1)))

