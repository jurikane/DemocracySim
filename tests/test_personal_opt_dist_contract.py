from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.ballots import ordering_from_distribution


def _assert_is_dist(x: np.ndarray, *, tol: float = 1e-6) -> None:
    assert x.ndim == 1
    assert np.all(np.isfinite(x))
    assert np.all(x >= -tol)
    s = float(x.sum())
    assert abs(s - 1.0) <= tol


def test_personal_opt_dist_is_valid_and_matches_personality_group_ordering() -> None:
    model, _cfg = create_test_model()
    agents = list(getattr(model, "voting_agents", []) or [])
    assert agents, "Expected test model to create voting_agents"

    for a in agents:
        if a is None:
            continue
        dist = np.asarray(getattr(a, "personal_opt_dist"))
        _assert_is_dist(dist)

        ordering = np.asarray(getattr(a, "personality_group"))
        implied = np.argsort(dist)[::-1]
        assert implied.shape == ordering.shape
        assert np.array_equal(implied, ordering), (
            f"personal_opt_dist ordering mismatch for agent {getattr(a, 'unique_id', '?')}: "
            f"implied={implied.tolist()} personality_group={ordering.tolist()}"
        )


def test_personal_opt_dist_ordering_from_distribution_matches_personality_group() -> None:
    """Contract: converting personal_opt_dist to an ordering reproduces personality_group."""
    model, _cfg = create_test_model(seed=124)
    agents = [a for a in (getattr(model, "voting_agents", []) or []) if a is not None]
    assert agents

    for a in agents:
        dist = np.asarray(getattr(a, "personal_opt_dist"), dtype=np.float32)
        ordering = np.asarray(getattr(a, "personality_group"), dtype=np.int64)
        derived = ordering_from_distribution(dist)
        assert np.array_equal(np.asarray(derived, dtype=np.int64), ordering)


def test_personal_opt_dist_is_deterministic_given_seed() -> None:
    # Determinism contract: with identical seed + config, personal_opt_dist must be identical.
    m1, _ = create_test_model(seed=123)
    m2, _ = create_test_model(seed=123)

    a1 = [a for a in (getattr(m1, "voting_agents", []) or []) if a is not None]
    a2 = [a for a in (getattr(m2, "voting_agents", []) or []) if a is not None]
    assert len(a1) == len(a2)

    # Match by agent_id to be robust to list ordering.
    by_id_1 = {int(a.unique_id): a for a in a1}
    by_id_2 = {int(a.unique_id): a for a in a2}
    assert by_id_1.keys() == by_id_2.keys()

    for aid in sorted(by_id_1.keys()):
        dx = np.asarray(getattr(by_id_1[aid], "personal_opt_dist"), dtype=np.float32)
        dy = np.asarray(getattr(by_id_2[aid], "personal_opt_dist"), dtype=np.float32)
        np.testing.assert_allclose(dx, dy, rtol=0, atol=0)


@pytest.fixture()
def v2_run_dir(tmp_path: Path) -> Path:
    """Run a tiny schema v2 headless run and return the run dir."""
    from src.config.loader import load_config
    from scripts.run_headless import run_once

    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config("test.yaml")
    run_dir = out_root / "run_0"
    run_once(run_id=0, cfg=cfg, out_dir=run_dir)
    return run_dir


def test_personal_opt_dist_is_written_to_static_json(v2_run_dir: Path) -> None:
    p = v2_run_dir / "static.json"
    assert p.exists(), f"Missing static.json in {v2_run_dir}"

    static = json.loads(p.read_text())
    assert "personal_opt_dist" in static, "static.json missing personal_opt_dist"

    pod = static["personal_opt_dist"]
    assert isinstance(pod, dict)
    assert pod, "personal_opt_dist dict is empty"

    # Validate shape for one agent
    any_agent_id, dist = next(iter(pod.items()))
    assert isinstance(any_agent_id, str)
    d = np.asarray(dist, dtype=np.float32)
    _assert_is_dist(d)
