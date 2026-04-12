from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_replay as run_replay


class _DummyServer:
    def __init__(self) -> None:
        self.open_browser: bool | None = None

    def launch(self, *, open_browser: bool) -> None:
        self.open_browser = bool(open_browser)


def test_demo_flag_uses_bundled_demo_run(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _DummyServer()
    called: dict[str, object] = {}

    monkeypatch.setattr(run_replay.AppConfig, "model_validate", staticmethod(lambda cfg: cfg))
    monkeypatch.setattr(run_replay, "normalize_selected_run_dir", lambda run_dir, *, action_label: run_dir)

    def _capture_make_server(appcfg, run_dir):
        called["run_dir"] = Path(run_dir)
        return server

    monkeypatch.setattr(run_replay, "make_replay_server", _capture_make_server)

    rc = run_replay.replay_main(["--demo", "--no-browser"])

    assert rc == 0
    assert called["run_dir"] == run_replay.DEMO_RUN_DIR
    assert server.open_browser is False


def test_explicit_run_dir_launches_with_browser_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _DummyServer()
    called: dict[str, object] = {}

    monkeypatch.setattr(run_replay, "normalize_selected_run_dir", lambda run_dir, *, action_label: run_dir)
    monkeypatch.setattr(run_replay.AppConfig, "model_validate", staticmethod(lambda cfg: cfg))

    def _capture_make_server(appcfg, run_dir):
        called["run_dir"] = Path(run_dir)
        return server

    monkeypatch.setattr(run_replay, "make_replay_server", _capture_make_server)

    rc = run_replay.replay_main([str(run_replay.DEMO_RUN_DIR)])

    assert rc == 0
    assert called["run_dir"] == run_replay.DEMO_RUN_DIR
    assert server.open_browser is True


def test_demo_and_run_dir_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        run_replay.replay_main(["--demo", str(run_replay.DEMO_RUN_DIR)])
