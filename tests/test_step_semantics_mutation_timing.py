from __future__ import annotations

from typing import Callable

import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_mutation_called_at_start_of_step_2_only() -> None:
    """Lock step semantics: mutation from step t is applied at the start of step t+1.

    We do *not* assert anything about whether mutation changes the grid (that depends
    on turnout and mu); we only assert call timing.
    """
    model, _cfg = create_test_model(seed=123, num_agents=30, num_colors=3, num_areas=2, max_steps=3, mu=0.2)

    calls: list[tuple[int, int]] = []

    for area in model.areas:
        orig: Callable[[], None] = area.mutate_cells

        def _wrapped_mutate(*, _orig=orig, _area_id=int(area.unique_id)) -> None:
            # scheduler.steps has already been incremented at this point.
            calls.append((int(model.scheduler.steps), _area_id))
            _orig()

        area.mutate_cells = _wrapped_mutate  # type: ignore[method-assign]

    # Step 1: no previous election => no mutation call
    model.step()
    assert calls == []

    # Step 2: mutation should be invoked once per area before elections run
    model.step()
    assert len(calls) == len(model.areas)
    assert all(step == 2 for step, _area_id in calls)

