from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import yaml

from src.config.loader import load_config, resolve_output_dir
from scripts.run_headless import run_once


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    overrides: dict[str, object]


def _set_nested_attr(obj, path: str, value) -> None:
    cur = obj
    parts = path.split(".")
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def _build_default_matrix(base_cfg) -> list[Scenario]:
    m = base_cfg.model
    fee_hi = min(0.10, max(0.0, float(getattr(m, "election_cost_rate", 0.02)) * 2.0))
    return [
        Scenario(
            name="static0_baseline",
            description="All votes self-regarding (altruism static=0).",
            overrides={
                "model.altruism_mode": "static",
                "model.altruism_static": 0.0,
            },
        ),
        Scenario(
            name="static1_baseline",
            description="All votes altruistic (altruism static=1).",
            overrides={
                "model.altruism_mode": "static",
                "model.altruism_static": 1.0,
            },
        ),
        Scenario(
            name="static0_fee0",
            description="Self-regarding voting, no election fee (isolates free-rider fee effect).",
            overrides={
                "model.altruism_mode": "static",
                "model.altruism_static": 0.0,
                "model.election_cost_rate": 0.0,
            },
        ),
        Scenario(
            name="static0_fee_high",
            description="Self-regarding voting, higher election fee (amplifies free-rider pressure).",
            overrides={
                "model.altruism_mode": "static",
                "model.altruism_static": 0.0,
                "model.election_cost_rate": fee_hi,
            },
        ),
        Scenario(
            name="static0_partalpha0",
            description="Self-regarding voting, participation learning disabled (alpha=0).",
            overrides={
                "model.altruism_mode": "static",
                "model.altruism_static": 0.0,
                "model.participation_alpha": 0.0,
            },
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small behavioral sanity matrix (non-DOE).")
    ap.add_argument("--config", "-c", default="doe.yaml", help="Base config path/name.")
    ap.add_argument("--steps", type=int, default=150, help="Override num_steps for all scenarios.")
    ap.add_argument("--seed", type=int, default=169, help="Base seed for comparability.")
    ap.add_argument("--rule-idx", type=int, default=1, help="Voting rule idx to test (default: approval=1).")
    ap.add_argument("--out-root", type=str, default=None, help="Optional explicit output root.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned scenarios only.")
    args = ap.parse_args()

    base = load_config(args.config)
    base = base.model_copy(deep=True)
    base.simulation.runs = 1
    base.simulation.num_steps = int(args.steps)
    base.simulation.base_seed = int(args.seed)
    base.simulation.store_grid = False
    base.simulation.grid_interval = 1
    base.model.rule_idx = int(args.rule_idx)

    scenarios = _build_default_matrix(base)

    if args.out_root:
        root = Path(args.out_root)
    else:
        base_out = resolve_output_dir(base)
        root = base_out / f"sanity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for i, sc in enumerate(scenarios):
        cfg = base.model_copy(deep=True)
        for path, value in sc.overrides.items():
            _set_nested_attr(cfg, path, value)
        sc_dir = root / f"{i:02d}_{sc.name}"
        sc_dir.mkdir(parents=True, exist_ok=True)
        cfg_dump = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else deepcopy(cfg)
        (sc_dir / "scenario_overrides.json").write_text(
            json.dumps({"name": sc.name, "description": sc.description, "overrides": sc.overrides}, indent=2),
            encoding="utf-8",
        )
        with (sc_dir / "config_used.yaml").open("w") as f:
            yaml.safe_dump(cfg_dump, f)
        manifest.append(
            {
                "scenario_idx": i,
                "scenario_name": sc.name,
                "description": sc.description,
                "run_dir": str(sc_dir / "run_0"),
                **{f"override__{k}": v for k, v in sc.overrides.items()},
            }
        )
        if args.dry_run:
            continue
        (sc_dir / "run_0").mkdir(parents=True, exist_ok=True)
        run_once(run_id=0, cfg=cfg, out_dir=(sc_dir / "run_0"))

    import pandas as pd

    pd.DataFrame(manifest).to_csv(root / "sanity_matrix_manifest.csv", index=False)
    print("Sanity matrix root:", root)
    print("Scenarios:", len(scenarios))
    if args.dry_run:
        print("Dry run only.")


if __name__ == "__main__":
    main()

