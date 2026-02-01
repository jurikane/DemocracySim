from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def _assert_is_dist(x: np.ndarray, *, tol: float = 1e-6) -> None:
    assert x.ndim == 1
    assert np.all(np.isfinite(x))
    assert np.all(x >= -tol)
    s = float(x.sum())
    assert abs(s - 1.0) <= tol


def test_personal_opt_dist_matches_personality_group_ordering() -> None:
    """Batch 0 contract: naming + invariant.

    - VoteAgent.personality_group is a ColorOrdering
    - VoteAgent.personality is a ColorDistribution
    - argsort(personality)[::-1] == personality_group
    """
    model, _cfg = create_test_model(seed=123)
    agents = [a for a in (getattr(model, "voting_agents", []) or []) if a is not None]
    assert agents

    for a in agents:
        dist = np.asarray(getattr(a, "personality"), dtype=np.float32)
        _assert_is_dist(dist)

        ordering = np.asarray(getattr(a, "personality_group"), dtype=np.int64)
        implied = np.argsort(dist)[::-1]
        assert implied.shape == ordering.shape
        assert np.array_equal(implied, ordering)


def test_personality_group_distribution_is_deterministic_given_seed() -> None:
    m1, _ = create_test_model(seed=123)
    m2, _ = create_test_model(seed=123)

    a1 = [a for a in (getattr(m1, "voting_agents", []) or []) if a is not None]
    a2 = [a for a in (getattr(m2, "voting_agents", []) or []) if a is not None]
    assert len(a1) == len(a2)

    by_id_1 = {int(a.unique_id): a for a in a1}
    by_id_2 = {int(a.unique_id): a for a in a2}
    assert by_id_1.keys() == by_id_2.keys()

    for aid in sorted(by_id_1.keys()):
        dx = np.asarray(by_id_1[aid].personality, dtype=np.float32)
        dy = np.asarray(by_id_2[aid].personality, dtype=np.float32)
        np.testing.assert_allclose(dx, dy, rtol=0, atol=0)
