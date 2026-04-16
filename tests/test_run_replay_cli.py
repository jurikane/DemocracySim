from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_replay as run_replay


class _DummyServer:
    def __init__(self) -> None:
        self.open_browser: bool | None = None

    def launch(self, *, open_browser: bool) -> None:
        self.open_browser = bool(open_browser)


def _write_fake_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.yaml").write_text("config_ref: cfg.yaml\n", encoding="utf-8")
    (run_dir / "cfg.yaml").write_text("{}\n", encoding="utf-8")
    return run_dir


def test_explicit_run_dir_launches_with_browser_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server = _DummyServer()
    called: dict[str, object] = {}
    run_dir = _write_fake_run_dir(tmp_path)

    monkeypatch.setattr(run_replay.AppConfig, "model_validate", staticmethod(lambda cfg: cfg))
    monkeypatch.setattr(run_replay, "normalize_selected_run_dir", lambda run_dir, *, action_label: run_dir)

    def _capture_make_server(appcfg, run_dir):
        called["run_dir"] = Path(run_dir)
        return server

    monkeypatch.setattr(run_replay, "make_replay_server", _capture_make_server)

    rc = run_replay.replay_main([str(run_dir)])

    assert rc == 0
    assert called["run_dir"] == run_dir
    assert server.open_browser is True


def test_interactive_picker_launches_selected_run_without_browser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server = _DummyServer()
    called: dict[str, object] = {}
    run_dir = _write_fake_run_dir(tmp_path)

    monkeypatch.setattr(run_replay, "pick_run_dir_interactive", lambda *, action_label: run_dir)
    monkeypatch.setattr(run_replay.AppConfig, "model_validate", staticmethod(lambda cfg: cfg))
    monkeypatch.setattr(run_replay, "normalize_selected_run_dir", lambda run_dir, *, action_label: run_dir)

    def _capture_make_server(appcfg, run_dir):
        called["run_dir"] = Path(run_dir)
        return server

    monkeypatch.setattr(run_replay, "make_replay_server", _capture_make_server)

    rc = run_replay.replay_main(["--no-browser"])

    assert rc == 0
    assert called["run_dir"] == run_dir
    assert server.open_browser is False


def test_no_selection_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_replay, "pick_run_dir_interactive", lambda *, action_label: None)

    rc = run_replay.replay_main([])

    assert rc == 1
