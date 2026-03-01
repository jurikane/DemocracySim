from __future__ import annotations

from src.replay.replay_server import _resolve_num_voters_per_area


def test_replay_static_uses_canonical_num_voters_per_area() -> None:
    static = {"num_voters_per_area": {"0": 10, "1": 12}}
    out = _resolve_num_voters_per_area(static)
    assert out == {"0": 10, "1": 12}


def test_replay_static_missing_voter_counts_returns_empty_mapping() -> None:
    out = _resolve_num_voters_per_area({})
    assert out == {}


def test_replay_static_rejects_legacy_voters_per_area_key() -> None:
    static = {"voters_per_area": {"0": 11}}
    try:
        _resolve_num_voters_per_area(static)
    except ValueError as exc:
        assert "voters_per_area" in str(exc)
    else:
        raise AssertionError("Expected ValueError for removed legacy key")


def test_replay_static_rejects_payloads_that_still_contain_legacy_key_even_with_canonical() -> None:
    static = {
        "num_voters_per_area": {"0": 7},
        "voters_per_area": {"0": 99},
    }
    try:
        _resolve_num_voters_per_area(static)
    except ValueError as exc:
        assert "voters_per_area" in str(exc)
    else:
        raise AssertionError("Expected ValueError when removed legacy key is present")


def test_replay_static_rejects_non_object_canonical_value() -> None:
    static = {"num_voters_per_area": 123}
    try:
        _resolve_num_voters_per_area(static)
    except ValueError as exc:
        assert "num_voters_per_area" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-object canonical key")


def test_replay_static_rejects_non_object_legacy_value() -> None:
    static = {"voters_per_area": 123}
    try:
        _resolve_num_voters_per_area(static)
    except ValueError as exc:
        assert "voters_per_area" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-object legacy key")
