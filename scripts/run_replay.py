"""Replay script: given a run directory produced by run_headless,
starts the replay server.

Usage: python -m scripts.run_replay <run_dir>

If <run_dir> is omitted, the script will list available runs under
<project_root>/data/simulation_output and prompt for a selection.
"""
import sys
from pathlib import Path
import yaml
from typing import Optional

from src.config.schema import AppConfig
from src.replay.replay_server import make_replay_server
from src.config.loader import get_project_root


def _default_runs_base_dir() -> Path:
    return get_project_root() / "data" / "simulation_output"


def _resolve_run_dir(arg: Optional[str]) -> Optional[Path]:
    """Resolve the run directory passed by the user.

    Resolution order for relative paths:
      1) as provided relative to CWD
      2) project-root-relative
      3) under <project_root>/data/simulation_output/<arg>
    """
    if not arg:
        return None

    p = Path(arg)
    if p.is_absolute():
        return p

    # 1) relative to CWD
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    # 2) relative to project root
    pr = get_project_root()
    if (pr / p).exists():
        return (pr / p).resolve()
    # 3) relative to default runs base
    base = _default_runs_base_dir()
    if (base / p).exists():
        return (base / p).resolve()

    # Return the most reasonable candidate (project root) for error reporting
    return (pr / p).resolve()


def _list_runs_under_timestamp(ts_dir: Path) -> list[Path]:
    """Return run directories (run_0, run_1, ...) under a timestamp folder."""
    runs = [p for p in ts_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name)
    return runs


def _pick_from_list(prompt: str, items: list[Path]) -> Optional[Path]:
    if not items:
        return None
    print(prompt)
    for i, p in enumerate(items, start=1):
        print(f"  {i}) {p.name}")
    while True:
        raw = input("\nEnter number (or 'q' to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= idx <= len(items):
            return items[idx - 1]
        print(f"Please enter a number between 1 and {len(items)}.")


def _pick_run_dir_interactive() -> Optional[Path]:
    """Ask the user to pick a run directory from data/simulation_output.

    Headless layout is:
      data/simulation_output/<timestamp>/run_<i>/...

    This picker first selects <timestamp>, then selects run_<i>.
    """
    base = _default_runs_base_dir()
    if not base.exists():
        print("No simulation_output directory found at:", base)
        return None

    timestamps = [p for p in base.iterdir() if p.is_dir()]
    timestamps.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not timestamps:
        print("No runs found in:", base)
        return None

    ts_dir = _pick_from_list( "No replay directory given. "
        "Which timestamp folder do you want to replay?\n", timestamps)
    if ts_dir is None:
        return None

    run_dirs = _list_runs_under_timestamp(ts_dir)
    if not run_dirs:
        print("No run_* folders found in:", ts_dir)
        return None
    if len(run_dirs) == 1:
        return run_dirs[0]

    return _pick_from_list("\nWhich run do you want to replay?\n", run_dirs)


def replay_main():
    run_dir: Optional[Path]
    if len(sys.argv) < 2:
        run_dir = _pick_run_dir_interactive()
    else:
        run_dir = _resolve_run_dir(sys.argv[1])

    if run_dir is None:
        print("Usage: python -m scripts.run_replay <run_dir>")
        return

    # If user passed a timestamp folder with multiple run_* folders, ask which one.
    if run_dir.exists() and run_dir.is_dir() and not (run_dir / "steps").exists():
        run_dirs = _list_runs_under_timestamp(run_dir)
        if len(run_dirs) > 1:
            picked = _pick_from_list(f"Multiple runs found in {run_dir.name}. Pick one:\n", run_dirs)
            if picked is None:
                return
            run_dir = picked
        elif len(run_dirs) == 1:
            run_dir = run_dirs[0]

    if not run_dir.exists():
        print("Run directory does not exist:", run_dir)
        return

    meta_path = run_dir / "meta.yaml"
    meta = None
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text())

    # Start replay server
    if meta is not None and "config_ref" in meta:
        try:
            cfg_ref = meta.get("config_ref")
            if not isinstance(cfg_ref, str) or not cfg_ref:
                raise ValueError("meta.yaml has invalid config_ref")
            cfg_path = (run_dir / cfg_ref).resolve()
            cfg = yaml.safe_load(cfg_path.read_text())
            appcfg = AppConfig.model_validate(cfg)
            server = make_replay_server(appcfg, run_dir)
            print("Starting replay server on http://127.0.0.1:8521 ...")
            server.launch(open_browser=True)
        except Exception as e:
            print("Could not start replay server:", e)
    else:
        print("meta.yaml missing or has no config_ref; cannot start server.")


if __name__ == '__main__':
    replay_main()
