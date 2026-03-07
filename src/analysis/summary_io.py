from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml

@dataclass(frozen=True)
class _RunMetaStaticParseResult:
    ok: bool
    meta: dict[str, Any] | None
    static: dict[str, Any] | None
    num_colors: int | None
    error: str | None

def _parse_required_run_meta_static(*, run_dir: Path) -> _RunMetaStaticParseResult:
    meta_path = run_dir / "meta.yaml"
    static_path = run_dir / "static.json"
    missing = [str(p) for p in (meta_path, static_path) if not p.exists()]
    if missing:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Missing required run metadata artifact(s): {', '.join(missing)}",
        )

    try:
        meta_raw = meta_path.read_text(encoding="utf-8")
    except OSError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to read meta.yaml at {meta_path}: {e}",
        )
    try:
        meta_obj = yaml.safe_load(meta_raw)
    except yaml.YAMLError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to parse YAML in {meta_path}: {e}",
        )
    if not isinstance(meta_obj, dict):
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid meta.yaml root in {meta_path}: expected mapping",
        )
    run_meta = meta_obj.get("run")
    if not isinstance(run_meta, dict):
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid or missing 'run' section in {meta_path}",
        )
    for req_key in ("run_seed", "rule_idx"):
        if req_key not in run_meta:
            return _RunMetaStaticParseResult(
                ok=False,
                meta=None,
                static=None,
                num_colors=None,
                error=f"Missing required meta.run field '{req_key}' in {meta_path}",
            )
    try:
        int(run_meta["run_seed"])
        int(run_meta["rule_idx"])
    except (TypeError, ValueError) as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid meta.run seed/rule fields in {meta_path}: {e}",
        )

    try:
        static_raw = static_path.read_text(encoding="utf-8")
    except OSError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to read static.json at {static_path}: {e}",
        )
    try:
        static_obj = json.loads(static_raw)
    except json.JSONDecodeError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to parse JSON in {static_path}: {e}",
        )
    if not isinstance(static_obj, dict):
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid static.json root in {static_path}: expected object",
        )

    if "num_colors" not in static_obj:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Missing required field 'num_colors' in {static_path}",
        )
    try:
        num_colors = int(static_obj["num_colors"])
    except (TypeError, ValueError) as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid static.json field 'num_colors' in {static_path}: {e}",
        )
    if num_colors <= 0:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid static.json field 'num_colors' in {static_path}: {num_colors}",
        )

    return _RunMetaStaticParseResult(
        ok=True,
        meta=dict(meta_obj),
        static=dict(static_obj),
        num_colors=num_colors,
        error=None,
    )

def _load_required_run_meta_static(*, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    parsed = _parse_required_run_meta_static(run_dir=run_dir)
    if not parsed.ok or parsed.meta is None or parsed.static is None or parsed.num_colors is None:
        raise RuntimeError(parsed.error or f"Failed to parse required run metadata for {run_dir}")
    return parsed.meta, parsed.static, int(parsed.num_colors)

@dataclass(frozen=True)
class _ModelCfgParseResult:
    ok: bool
    model_cfg: dict[str, Any] | None
    source: Path | None
    error: str | None

def _parse_model_cfg_for_run(*, run_dir: Path) -> _ModelCfgParseResult:
    """Typed parse result for model section in config_used.yaml."""
    candidates = [
        run_dir / "config_used.yaml",
        run_dir.parent / "config_used.yaml",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        tried = ", ".join(str(p) for p in candidates)
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=None,
            error=f"Missing config_used.yaml (tried: {tried})",
        )

    cfg_path = existing[0]
    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
    except OSError as e:
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Failed to read config file {cfg_path}: {e}",
        )

    try:
        cfg = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Failed to parse YAML in {cfg_path}: {e}",
        )

    if not isinstance(cfg, dict):
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Invalid config root in {cfg_path}: expected mapping",
        )

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Invalid or missing 'model' section in {cfg_path}",
        )

    return _ModelCfgParseResult(
        ok=True,
        model_cfg=dict(model_cfg),
        source=cfg_path,
        error=None,
    )

def _load_model_cfg_for_run(*, run_dir: Path) -> dict[str, Any]:
    """Load model section from config_used.yaml (strict fail-fast)."""
    parsed = _parse_model_cfg_for_run(run_dir=run_dir)
    if not parsed.ok or parsed.model_cfg is None:
        raise RuntimeError(parsed.error or "Failed to parse model config")
    return parsed.model_cfg

def _load_required_finite_float_for_run(*, run_dir: Path, field: str) -> float:
    """Load required model float field from config_used.yaml (strict)."""
    model_cfg = _load_model_cfg_for_run(run_dir=run_dir)
    if field not in model_cfg:
        raise RuntimeError(f"Missing required model config field '{field}' in config_used.yaml")
    raw = model_cfg[field]
    try:
        val = float(raw)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Invalid model config field '{field}': {raw!r}") from e
    if not np.isfinite(val):
        raise RuntimeError(f"Invalid non-finite model config field '{field}': {raw!r}")
    return val

def _load_required_bool_for_run(*, run_dir: Path, field: str) -> bool:
    """Load required model bool field from config_used.yaml (strict)."""
    model_cfg = _load_model_cfg_for_run(run_dir=run_dir)
    if field not in model_cfg:
        raise RuntimeError(f"Missing required model config field '{field}' in config_used.yaml")
    raw = model_cfg[field]
    if not isinstance(raw, bool):
        raise RuntimeError(f"Invalid model config field '{field}': expected bool, got {type(raw).__name__}")
    return raw

def _load_participation_alpha_for_run(*, run_dir: Path) -> float:
    """Load model.participation_alpha from config_used.yaml (strict)."""
    return _load_required_finite_float_for_run(run_dir=run_dir, field="participation_alpha")

def _load_altruism_alpha_for_run(*, run_dir: Path) -> float:
    """Load model.altruism_alpha from config_used.yaml (strict)."""
    return _load_required_finite_float_for_run(run_dir=run_dir, field="altruism_alpha")

def _load_participation_signal_mode_for_run(*, run_dir: Path) -> str:
    """Load model.participation_signal_mode from config_used.yaml (strict)."""
    model_cfg = _load_model_cfg_for_run(run_dir=run_dir)
    field = "participation_signal_mode"
    if field not in model_cfg:
        raise RuntimeError(f"Missing required model config field '{field}' in config_used.yaml")
    raw = model_cfg[field]
    if not isinstance(raw, str):
        raise RuntimeError(f"Invalid model config field '{field}': expected str, got {type(raw).__name__}")
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Invalid model config field '{field}': empty string")
    return value

def _load_participation_signal_group_shrink_k_for_run(*, run_dir: Path) -> float:
    """Load model.participation_signal_group_shrink_k from config_used.yaml (strict)."""
    return _load_required_finite_float_for_run(run_dir=run_dir, field="participation_signal_group_shrink_k")

def _load_altruism_learning_for_run(*, run_dir: Path) -> bool:
    """Load model.altruism_learning from config_used.yaml (strict)."""
    return _load_required_bool_for_run(run_dir=run_dir, field="altruism_learning")

def _load_grid_with_carry_forward(*, run_dir: Path, step: int, max_step: int) -> np.ndarray | None:
    grids_dir = run_dir / "grids"
    if not grids_dir.exists():
        return None
    pad = len(str(int(max_step)))
    target = grids_dir / f"grid_{int(step):0{pad}d}.npy"
    if target.exists():
        return np.asarray(np.load(target))
    # Carry-forward: latest available <= step
    candidates: list[int] = []
    for p in grids_dir.glob("grid_*.npy"):
        stem = p.stem
        raw = stem.replace("grid_", "")
        try:
            idx = int(raw)
        except ValueError:
            continue
        if idx <= int(step):
            candidates.append(idx)
    if not candidates:
        return None
    chosen = max(candidates)
    chosen_path = grids_dir / f"grid_{int(chosen):0{pad}d}.npy"
    if not chosen_path.exists():
        return None
    return np.asarray(np.load(chosen_path))

def _load_area_agent_ids_from_static_overlays(*, run_dir: Path) -> dict[int, list[int]]:
    """Recover resident area->agent ids from typed static overlay artifacts."""
    static_path = run_dir / "static.json"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing static.json for typed overlay lookup: {run_dir}")

    try:
        static = json.loads(static_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"Failed reading static.json for typed overlays: {static_path}") from e

    artifacts = static.get("artifacts") if isinstance(static.get("artifacts"), dict) else {}
    rel = artifacts.get("cell_agents")
    if not isinstance(rel, str) or rel.strip() == "":
        raise KeyError(f"static.json missing required artifacts.cell_agents: {static_path}")
    cell_agents_path = run_dir / rel
    if not cell_agents_path.exists():
        raise FileNotFoundError(f"Missing typed cell_agents artifact: {cell_agents_path}")

    try:
        df = pd.read_parquet(cell_agents_path)
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"Failed reading typed cell_agents artifact: {cell_agents_path}") from e

    if not {"area_id", "agent_id"}.issubset(df.columns):
        raise KeyError(f"{cell_agents_path.name} missing required columns ['area_id', 'agent_id']")

    out: dict[int, set[int]] = {}
    for r in df[["area_id", "agent_id"]].dropna().itertuples(index=False):
        area_id = int(r.area_id)
        agent_id = int(r.agent_id)
        if area_id < 0 or agent_id < 0:
            continue
        if area_id not in out:
            out[area_id] = set()
        out[area_id].add(agent_id)
    return {k: sorted(v) for k, v in sorted(out.items())}
