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
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_one_area(**overrides):
    base = dict(
        seed=300,
        num_colors=2,
        num_personality_groups=2,
        num_agents=6,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=0,
        max_steps=3,
        # isolate from reward dynamics
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
        mu=0.25,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    # Ensure deterministic participation/voting so turnout>0.
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()
    return model


def _set_all_area_cells(area, color: int) -> None:
    for c in area.cells:
        c.color = int(color)
    area.update_color_distribution()


def test_mu_oracle_zero_means_no_cells_change_on_mutation_round() -> None:
    """Oracle: mu=0 => n_to_mutate==0 => mutation round performs no changes."""
    model = _model_one_area(mu=0.0)
    area = model.areas[0]

    # Step 1 establishes turnout>0 so mutate_cells would run on step 2 if mu>0.
    model.step()

    _set_all_area_cells(area, 1)
    colors_before = [int(c.color) for c in area.cells]

    model.step()  # step2: mutation happens at start, but mu=0 => no change

    colors_after = [int(c.color) for c in area.cells]
    assert colors_after == colors_before


def test_mu_oracle_one_means_all_cells_mutate_when_turnout_positive() -> None:
    """Oracle: mu=1 => all cells are selected for mutation (given turnout>0)."""
    model = _model_one_area(mu=1.0)
    area = model.areas[0]

    # Make mutation deterministic: always pick voted_ordering[0] (color 0).
    model.color_probs = np.asarray([1.0, 0.0], dtype=np.float64)

    model.step()  # establish turnout>0 and voted_ordering
    # Ensure mutation draws always map to color 0 (voted_ordering[0]).
    area._voted_ordering = np.asarray([0, 1], dtype=np.int64)
    _set_all_area_cells(area, 1)  # start from all 1 so mutations are countable

    model.step()  # step2 applies mutation from step1

    assert all(int(c.color) == 0 for c in area.cells)


def test_mu_metamorphic_increasing_mu_increases_mutated_cell_count() -> None:
    """Metamorphic: with a deterministic sampler and deterministic chosen color, larger mu mutates >= cells."""
    mu1 = 0.25
    mu2 = 0.5
    assert mu2 > mu1

    m1 = _model_one_area(seed=301, mu=mu1)
    m2 = _model_one_area(seed=301, mu=mu2)
    a1 = m1.areas[0]
    a2 = m2.areas[0]

    # Deterministic mutation: always choose color 0.
    m1.color_probs = np.asarray([1.0, 0.0], dtype=np.float64)
    m2.color_probs = np.asarray([1.0, 0.0], dtype=np.float64)

    # Deterministic selection of cells to mutate: always take the first k cells.
    def _sample(seq, k):  # type: ignore[no-untyped-def]
        return list(seq)[: int(k)]

    m1.random.sample = _sample  # type: ignore[method-assign]
    m2.random.sample = _sample  # type: ignore[method-assign]

    m1.step()
    m2.step()
    _set_all_area_cells(a1, 1)
    _set_all_area_cells(a2, 1)

    m1.step()
    m2.step()

    num0_1 = sum(1 for c in a1.cells if int(c.color) == 0)
    num0_2 = sum(1 for c in a2.cells if int(c.color) == 0)
    assert num0_2 >= num0_1

    # Exact expected counts given deterministic sampling and all-1 start state:
    expected1 = int(mu1 * a1.num_cells)
    expected2 = int(mu2 * a2.num_cells)
    assert num0_1 == expected1
    assert num0_2 == expected2


def test_mu_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"mu must be in \[0,1\]"):
        _model_one_area(mu=-0.01)
    with pytest.raises(ValueError, match=r"mu must be in \[0,1\]"):
        _model_one_area(mu=1.01)
