from __future__ import annotations

import warnings

from src.replay.replay_server import _resolve_num_voters_per_area


def test_replay_static_uses_canonical_num_voters_per_area_without_warning() -> None:
    static = {"num_voters_per_area": {"0": 10, "1": 12}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _resolve_num_voters_per_area(static)
    assert out == {"0": 10, "1": 12}
    assert len(caught) == 0


def test_replay_static_accepts_legacy_voters_per_area_with_deprecation_warning() -> None:
    static = {"voters_per_area": {"0": 11}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _resolve_num_voters_per_area(static)
    assert out == {"0": 11}
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "voters_per_area" in str(caught[0].message)


def test_replay_static_prefers_canonical_key_when_both_present_and_warns() -> None:
    static = {
        "num_voters_per_area": {"0": 7},
        "voters_per_area": {"0": 99},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _resolve_num_voters_per_area(static)
    assert out == {"0": 7}
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any(issubclass(w.category, UserWarning) for w in caught)

