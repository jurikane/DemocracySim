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


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "steps.parquet").exists()
        or (path / "static.json").exists()
    )


def list_runs_under_timestamp(ts_dir: Path) -> list[Path]:
    runs = [p for p in ts_dir.iterdir() if _is_run_dir(p) and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name)
    return runs


def list_run_dirs_recursive(root: Path) -> list[Path]:
    runs = [p for p in root.rglob("run_*") if _is_run_dir(p)]
    runs.sort(key=lambda p: p.as_posix())
    return runs


def _list_subdirs(path: Path) -> list[Path]:
    subdirs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith("run_")]
    subdirs.sort(key=lambda p: p.name)
    return subdirs


def pick_from_list(prompt: str, items: list[Path], *, base: Optional[Path] = None) -> Optional[Path]:
    if not items:
        return None
    print(prompt)
    for i, p in enumerate(items, start=1):
        label = p.name
        if base is not None:
            try:
                label = str(p.relative_to(base))
            except ValueError:
                label = p.name
        print(f"  {i}) {label}")
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
        return pick_run_dir_through_folders(ts_dir, action_label=action_label)
    if len(run_dirs) == 1:
        return run_dirs[0]
    return pick_from_list(f"\nWhich run do you want to {action_label}?\n", run_dirs)


def pick_run_dir_through_folders(start_dir: Path, *, action_label: str) -> Optional[Path]:
    current = start_dir
    while True:
        run_dirs = list_runs_under_timestamp(current)
        if run_dirs:
            if len(run_dirs) == 1:
                return run_dirs[0]
            return pick_from_list(f"\nWhich run do you want to {action_label}?\n", run_dirs)

        subdirs = _list_subdirs(current)
        if not subdirs:
            print("No run_* folders found in:", current)
            return None

        next_dir = pick_from_list(
            f"No run_* folders found in {current.name}. Pick a subfolder to continue:\n",
            subdirs,
        )
        if next_dir is None:
            return None
        current = next_dir


def normalize_selected_run_dir(run_dir: Path, *, action_label: str) -> Optional[Path]:
    """If path points to timestamp dir, resolve/pick concrete run_x dir."""
    if run_dir.exists() and run_dir.is_dir() and not (run_dir / "steps.parquet").exists():
        run_dirs = list_runs_under_timestamp(run_dir)
        if not run_dirs:
            return pick_run_dir_through_folders(run_dir, action_label=action_label)
        if len(run_dirs) > 1:
            picked = pick_from_list(
                f"Multiple runs found in {run_dir.name}. Pick one to {action_label}:\n",
                run_dirs,
                base=run_dir,
            )
            if picked is None:
                return None
            return picked
        if len(run_dirs) == 1:
            return run_dirs[0]
    return run_dir

