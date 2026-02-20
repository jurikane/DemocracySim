from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        # Valid ScoreVector in [0,1].
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _force_fixed_voting_rule(model) -> None:
    """Deterministic rule: pick option 0, then ascending order."""
    m = int(np.asarray(model.options).shape[0])
    ordering = np.arange(m, dtype=np.int64)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return ordering

    model.voting_rule = _rule


def _make_model_one_area_for_altruism(**overrides):
    base = dict(
        seed=200,
        num_agents=2,
        num_colors=2,
        num_personality_groups=2,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=0,  # avoid RNG usage in knowledge sampling
        mu=0.0,  # avoid mutation effects
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
        max_steps=10,
        altruism_learning=True,
        altruism_init=0.5,
        altruism_alpha=0.2,
        altruism_clip_min=0.0,
        altruism_clip_max=10.0,
        satisfaction_mode="area",
        satisfaction_baseline_alpha=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    _force_fixed_voting_rule(model)
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()
    return model


def _set_all_cells(area, color: int) -> None:
    for c in area.cells:
        c.color = int(color)
    area.update_color_distribution()


def test_altruism_learning_interaction_baseline_alpha_controls_persistence() -> None:
    """Interaction: satisfaction_baseline_alpha controls whether altruism keeps updating
    when the satisfaction value stays at its new level.

    We patch satisfaction value to follow the sequence:
      step1: sv=0 (baseline init, signal=0)
      step2: sv=1 (signal=+1)
      step3: sv=1 (signal depends on baseline_alpha)

    With baseline_alpha=1:
      baseline becomes 1 at step2 => step3 signal=0 => 1 update total.
    With baseline_alpha=0:
      baseline stays 0 => step3 signal=+1 => 2 updates total.
    """
    alpha = 0.2
    init_a = 0.5

    m_fast = _make_model_one_area_for_altruism(
        seed=201,
        altruism_alpha=alpha,
        altruism_init=init_a,
        satisfaction_baseline_alpha=1.0,
        max_steps=3,
    )
    m_slow = _make_model_one_area_for_altruism(
        seed=201,
        altruism_alpha=alpha,
        altruism_init=init_a,
        satisfaction_baseline_alpha=0.0,
        max_steps=3,
    )

    def _sv(_self, *, area, model) -> float:  # type: ignore[no-untyped-def]
        # scheduler.steps is 1 on first recorded step.
        s = int(model.scheduler.steps)
        if s <= 1:
            return 0.0
        return 1.0

    for m in (m_fast, m_slow):
        for a in m.voting_agents:
            if a is None:
                continue
            a.compute_dissatisfaction_value = _sv.__get__(a, type(a))  # bind method

    a_fast = m_fast.voting_agents[0]
    a_slow = m_slow.voting_agents[0]
    assert a_fast is not None and a_slow is not None

    for _ in range(3):
        m_fast.step()
        m_slow.step()

    assert float(a_fast.altruism_factor) == pytest.approx(init_a + alpha * 1.0, abs=1e-12)
    assert float(a_slow.altruism_factor) == pytest.approx(init_a + alpha * 2.0, abs=1e-12)


def test_altruism_learning_interaction_satisfaction_mode_changes_update_when_global_constant() -> None:
    """Interaction: if global distribution is constant across steps but local area flips,
    then satisfaction_mode='global' yields no altruism update, while 'area' yields an update.

    Construction:
    - two areas, swap their colors between step1 and step2 => global mean stays [0.5,0.5]
    - agent personality fixed at [1,0]
    - satisfaction_baseline_alpha=0 => baseline fixed at step1 sv
    """
    cfg = dict(
        seed=202,
        num_agents=50,
        num_colors=2,
        num_personality_groups=2,
        num_areas=2,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=5,
        area_size_variance=0.0,
        known_cells=0,
        mu=0.0,
        max_steps=2,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
        altruism_learning=True,
        altruism_init=0.5,
        altruism_alpha=0.25,
        altruism_clip_min=0.0,
        altruism_clip_max=10.0,
        satisfaction_baseline_alpha=0.0,
    )

    m_area, _ = create_test_model(**cfg, satisfaction_mode="area")
    m_glob, _ = create_test_model(**cfg, satisfaction_mode="global")
    for m in (m_area, m_glob):
        _force_fixed_voting_rule(m)
        for a in m.voting_agents:
            if a is None:
                continue
            a.participation_strategy = _AlwaysParticipate()
            a.voting_strategy = _ZeroBallot()

    # Pick a focal agent in area0 and set personality to [1,0].
    area0_a, area1_a = m_area.areas[0], m_area.areas[1]
    area0_g, area1_g = m_glob.areas[0], m_glob.areas[1]
    agent_a = area0_a.agents[0]
    agent_g = area0_g.agents[0]
    agent_a.personal_opt_dist = np.asarray([1.0, 0.0], dtype=np.float64)
    agent_g.personal_opt_dist = np.asarray([1.0, 0.0], dtype=np.float64)

    # Step 1 colors: area0 all 0, area1 all 1.
    _set_all_cells(area0_a, 0)
    _set_all_cells(area1_a, 1)
    _set_all_cells(area0_g, 0)
    _set_all_cells(area1_g, 1)
    m_area.update_global_color_distribution()
    m_glob.update_global_color_distribution()

    # Step 1 runs (baseline init).
    m_area.step()
    m_glob.step()

    # Step 2: swap colors so global stays constant but local flips.
    _set_all_cells(area0_a, 1)
    _set_all_cells(area1_a, 0)
    _set_all_cells(area0_g, 1)
    _set_all_cells(area1_g, 0)
    # Let the models refresh global if they need it.
    m_area.update_global_color_distribution()
    m_glob.update_global_color_distribution()

    m_area.step()
    m_glob.step()

    # Global-mode agent should see constant sv => signal 0 => no altruism change.
    assert float(agent_g.altruism_factor) == pytest.approx(0.5, abs=1e-12)
    # Area-mode agent sees sv increase from 0 to 1 => signal +1 => altruism increases by alpha.
    assert float(agent_a.altruism_factor) == pytest.approx(0.5 + 0.25 * 1.0, abs=1e-12)


def test_altruism_learning_interaction_clip_applies_in_pipeline() -> None:
    """Interaction: a large positive signal updates altruism but is clipped by clip_max."""
    alpha = 1.0
    init_a = 0.5
    clip_max = 0.6
    model = _make_model_one_area_for_altruism(
        seed=203,
        altruism_alpha=alpha,
        altruism_init=init_a,
        altruism_clip_min=0.0,
        altruism_clip_max=clip_max,
        satisfaction_baseline_alpha=0.0,
        max_steps=2,
    )

    def _sv(_self, *, area, model) -> float:  # type: ignore[no-untyped-def]
        # step1: 0, step2: 1 => signal +1 on step2
        return 0.0 if int(model.scheduler.steps) <= 1 else 1.0

    for a in model.voting_agents:
        if a is None:
            continue
        a.compute_dissatisfaction_value = _sv.__get__(a, type(a))

    focal = model.voting_agents[0]
    assert focal is not None

    model.step()  # step1 baseline init
    model.step()  # step2 update, should clip

    assert float(focal.altruism_factor) == pytest.approx(clip_max, abs=1e-12)

