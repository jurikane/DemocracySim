"""Replay script: given a run directory produced by run_headless,
starts the replay server.

Usage: python -m scripts.run_replay <run_dir>

If <run_dir> is omitted, the script will list available runs under
<project_root>/data/simulation_output and prompt for a selection.
"""
import argparse
from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError
import yaml

from src.config.loader import get_project_root
from src.config.schema import AppConfig
from src.replay.replay_server import make_replay_server
from src.utils.run_path_picker import (
    normalize_selected_run_dir,
    pick_run_dir_interactive,
    resolve_run_dir,
)

DEMO_RUN_DIR = get_project_root() / "examples" / "demo_runs" / "approval_sparse_v1" / "run_0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a stored DemocracySim run")
    parser.add_argument("run_dir", nargs="?", help="Path to a run directory")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Replay the bundled public demo run",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab on launch",
    )
    return parser


def _resolve_requested_run_dir(run_dir_arg: Optional[str], *, use_demo: bool) -> Optional[Path]:
    if use_demo:
        return DEMO_RUN_DIR
    if run_dir_arg is None:
        run_dir = pick_run_dir_interactive(action_label="replay")
        return run_dir
    return resolve_run_dir(run_dir_arg)


def replay_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.demo and args.run_dir is not None:
        parser.error("argument --demo: not allowed with a run_dir")

    run_dir = _resolve_requested_run_dir(args.run_dir, use_demo=bool(args.demo))
    if run_dir is None:
        parser.print_usage()
        return 1

    normalized = normalize_selected_run_dir(run_dir, action_label="replay")
    if normalized is None:
        return 1
    run_dir = normalized

    if not run_dir.exists():
        print("Run directory does not exist:", run_dir)
        return 1

    meta_path = run_dir / "meta.yaml"
    meta = None
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    # Start replay server
    if meta is not None and "config_ref" in meta:
        try:
            cfg_ref = meta.get("config_ref")
            if not isinstance(cfg_ref, str) or not cfg_ref:
                raise ValueError("meta.yaml has invalid config_ref")
            cfg_path = (run_dir / cfg_ref).resolve()
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            appcfg = AppConfig.model_validate(cfg)
            server = make_replay_server(appcfg, run_dir)
            print("Starting replay server on http://127.0.0.1:8521 ...")
            server.launch(open_browser=not args.no_browser)
            return 0
        except (OSError, yaml.YAMLError, ValidationError, ValueError, RuntimeError) as e:
            print("Could not start replay server:", e)
            return 1
    else:
        print("meta.yaml missing or has no config_ref; cannot start server.")
        return 1


if __name__ == '__main__':
    raise SystemExit(replay_main())
