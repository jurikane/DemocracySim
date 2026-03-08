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
from pydantic import ValidationError

from src.config.schema import AppConfig
from src.replay.replay_server import make_replay_server
from src.utils.run_path_picker import (
    normalize_selected_run_dir,
    pick_run_dir_interactive,
    resolve_run_dir,
)


def replay_main():
    run_dir: Optional[Path]
    if len(sys.argv) < 2:
        run_dir = pick_run_dir_interactive(action_label="replay")
    else:
        run_dir = resolve_run_dir(sys.argv[1])

    if run_dir is None:
        print("Usage: python -m scripts.run_replay <run_dir>")
        return

    normalized = normalize_selected_run_dir(run_dir, action_label="replay")
    if normalized is None:
        return
    run_dir = normalized

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
        except (OSError, yaml.YAMLError, ValidationError, ValueError, RuntimeError) as e:
            print("Could not start replay server:", e)
    else:
        print("meta.yaml missing or has no config_ref; cannot start server.")


if __name__ == '__main__':
    replay_main()
