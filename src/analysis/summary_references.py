from __future__ import annotations

from math import factorial
from pathlib import Path
from typing import Any
import hashlib
import itertools
import json
import warnings

import numpy as np

from src.analysis.reference_benchmarks import (
    egalitarian_refs_mean_plus_lambda_gini,
    l1_dist,
    nash_ref_kl,
    rawlsian_ref_minimax_l2sq,
    utilitarian_ref_l2sq,
)
from src.analysis.summary_series import _personal_dists_from_static

SUMMARY_MODE_FULL = "full"
SUMMARY_MODE_FAST = "fast"
_SUMMARY_MODES = {SUMMARY_MODE_FULL, SUMMARY_MODE_FAST}


def _validate_summary_mode(mode: str) -> None:
    if str(mode) not in _SUMMARY_MODES:
        raise ValueError(f"Unsupported summary mode '{mode}'. Expected one of {_SUMMARY_MODES}.")

def _ref_cache_path(out_dir: Path, mode: str) -> Path:
    return out_dir / f"reference_cache_{mode}.json"

def _reference_cache_key(
    *,
    static: dict[str, Any],
    num_colors: int,
    area_agent_ids: dict[int, list[int]],
    mode: str,
) -> str:
    payload = {
        "version": 1,
        "mode": mode,
        "num_colors": int(num_colors),
        "personal_opt_dist": static.get("personal_opt_dist", {}),
        "area_agent_ids": {str(int(k)): [int(v) for v in vals] for k, vals in sorted(area_agent_ids.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _serialize_refs(refs: dict[str, np.ndarray | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in refs.items():
        out[k] = None if v is None else np.asarray(v, dtype=np.float64).tolist()
    return out

def _deserialize_refs(raw: dict[str, Any]) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        else:
            arr = np.asarray(v, dtype=np.float64)
            out[k] = arr if arr.ndim == 1 else None
    return out

def _compute_reference_set_for_dists(
    *,
    dists: np.ndarray,
    mode: str,
) -> dict[str, np.ndarray | None]:
    if dists.size == 0:
        return {
            "dist_to_ref_utilitarian": None,
            "dist_to_ref_nash": None,
            "dist_to_ref_rawlsian": None,
            "dist_to_ref_egalitarian": None,
            "dist_to_ref_egalitarian_lam025": None,
            "dist_to_ref_egalitarian_lam400": None,
        }

    util = utilitarian_ref_l2sq(dists)
    nash = nash_ref_kl(dists)
    if mode == SUMMARY_MODE_FAST:
        return {
            "dist_to_ref_utilitarian": np.asarray(util, dtype=np.float64),
            "dist_to_ref_nash": np.asarray(nash, dtype=np.float64),
            "dist_to_ref_rawlsian": None,
            "dist_to_ref_egalitarian": None,
            "dist_to_ref_egalitarian_lam025": None,
            "dist_to_ref_egalitarian_lam400": None,
        }

    rawl = rawlsian_ref_minimax_l2sq(dists)
    egal_refs = egalitarian_refs_mean_plus_lambda_gini(dists)
    return {
        "dist_to_ref_utilitarian": np.asarray(util, dtype=np.float64),
        "dist_to_ref_nash": np.asarray(nash, dtype=np.float64),
        "dist_to_ref_rawlsian": np.asarray(rawl, dtype=np.float64),
        "dist_to_ref_egalitarian": np.asarray(egal_refs.lam_mid, dtype=np.float64),
        "dist_to_ref_egalitarian_lam025": np.asarray(egal_refs.lam_low, dtype=np.float64),
        "dist_to_ref_egalitarian_lam400": np.asarray(egal_refs.lam_high, dtype=np.float64),
    }

def _load_or_compute_reference_payload(
    *,
    out_dir: Path,
    static: dict[str, Any],
    num_colors: int,
    area_agent_ids: dict[int, list[int]],
    mode: str,
    use_cache: bool,
) -> dict[str, Any]:
    _validate_summary_mode(mode)
    key = _reference_cache_key(
        static=static,
        num_colors=num_colors,
        area_agent_ids=area_agent_ids,
        mode=mode,
    )
    cache_path = _ref_cache_path(out_dir=out_dir, mode=mode)

    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("cache_key") == key
                and isinstance(cached.get("global"), dict)
                and isinstance(cached.get("areas"), dict)
            ):
                return {
                    "global": _deserialize_refs(cached["global"]),
                    "areas": {
                        int(k): _deserialize_refs(v)
                        for k, v in cached["areas"].items()
                        if isinstance(v, dict)
                    },
                }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Reference cache read failed at {cache_path}: {exc}. Recomputing reference payload.",
                RuntimeWarning,
                stacklevel=2,
            )

    d_global = _personal_dists_from_static(static=static, num_colors=num_colors)
    refs_global = _compute_reference_set_for_dists(dists=d_global, mode=mode)

    personal_by_id = static.get("personal_opt_dist", {}) if isinstance(static.get("personal_opt_dist"), dict) else {}
    refs_by_area: dict[int, dict[str, np.ndarray | None]] = {}
    for area_id, ids in area_agent_ids.items():
        rows: list[np.ndarray] = []
        for aid in ids:
            arr = np.asarray(personal_by_id.get(str(int(aid))), dtype=np.float64)
            if arr.ndim == 1 and arr.size == int(num_colors):
                s = float(np.sum(arr))
                if s > 0.0:
                    rows.append(arr / s)
        d_area = np.vstack(rows).astype(np.float64) if rows else np.asarray([], dtype=np.float64)
        refs_by_area[int(area_id)] = _compute_reference_set_for_dists(dists=d_area, mode=mode)

    payload = {
        "cache_key": key,
        "global": _serialize_refs(refs_global),
        "areas": {str(int(k)): _serialize_refs(v) for k, v in refs_by_area.items()},
    }
    if use_cache:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"global": refs_global, "areas": refs_by_area}
