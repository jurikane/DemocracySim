from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config.loader import get_project_root


def default_runs_base_dir() -> Path:
    return get_project_root() / "data" / "simulation_output"


def resolve_run_dir(arg: Optional[str]) -> Optional[Path]:
    """Resolve run/timestamp path from user input."""
    if not arg:
        return None

    p = Path(arg)
    if p.is_absolute():
        return p

    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()

    pr = get_project_root()
    if (pr / p).exists():
        return (pr / p).resolve()

    base = default_runs_base_dir()
    if (base / p).exists():
        return (base / p).resolve()

    return (pr / p).resolve()


def list_runs_under_timestamp(ts_dir: Path) -> list[Path]:
    runs = [p for p in ts_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name)
    return runs


def pick_from_list(prompt: str, items: list[Path]) -> Optional[Path]:
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


def pick_run_dir_interactive(*, action_label: str) -> Optional[Path]:
    """Pick run dir from default simulation output tree."""
    base = default_runs_base_dir()
    if not base.exists():
        print("No simulation_output directory found at:", base)
        return None

    timestamps = [p for p in base.iterdir() if p.is_dir()]
    timestamps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not timestamps:
        print("No runs found in:", base)
        return None

    ts_dir = pick_from_list(
        f"No run directory given. Which timestamp folder do you want to {action_label}?\n",
        timestamps,
    )
    if ts_dir is None:
        return None

    run_dirs = list_runs_under_timestamp(ts_dir)
    if not run_dirs:
        print("No run_* folders found in:", ts_dir)
        return None
    if len(run_dirs) == 1:
        return run_dirs[0]
    return pick_from_list(f"\nWhich run do you want to {action_label}?\n", run_dirs)


def normalize_selected_run_dir(run_dir: Path, *, action_label: str) -> Optional[Path]:
    """If path points to timestamp dir, resolve/pick concrete run_x dir."""
    if run_dir.exists() and run_dir.is_dir() and not (run_dir / "steps.parquet").exists():
        run_dirs = list_runs_under_timestamp(run_dir)
        if len(run_dirs) > 1:
            picked = pick_from_list(f"Multiple runs found in {run_dir.name}. Pick one to {action_label}:\n", run_dirs)
            if picked is None:
                return None
            return picked
        if len(run_dirs) == 1:
            return run_dirs[0]
    return run_dir

