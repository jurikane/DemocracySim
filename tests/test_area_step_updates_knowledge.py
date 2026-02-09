from __future__ import annotations

from tests.factory import create_test_model


def test_area_step_updates_known_cells_for_all_agents(monkeypatch) -> None:
    model, _ = create_test_model(num_agents=3, num_areas=1, num_colors=3)
    area = model.areas[0]

    calls = []

    def _spy_update_known_cells(self, *, area):  # type: ignore[no-untyped-def]
        calls.append(self.unique_id)

    for agent in area.agents:
        monkeypatch.setattr(agent, "update_known_cells", _spy_update_known_cells.__get__(agent, type(agent)))

    monkeypatch.setattr(area, "conduct_election", lambda: None)
    monkeypatch.setattr(area, "mutate_cells", lambda: None)

    area.step()

    assert sorted(calls) == sorted(a.unique_id for a in area.agents)
