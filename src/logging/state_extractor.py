"""State extraction utilities (Phase 1 implementation).

Contract:
- extract_state(model) -> dict with keys:
  'agent_features': np.ndarray shape (num_agents, F_a)
  'area_features': np.ndarray shape (num_areas, F_area)
  'grid_tensor': np.ndarray shape (H, W)
- dtype float32 for feature arrays, uint8 for grid.

agent_features columns:
[row, col, assets, num_elections_participated, personality_idx, confidence]
area_features columns:
[num_agents, num_cells, voter_turnout]
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


def _snapshot_grid(model) -> np.ndarray:
    cells = getattr(model, 'color_cells', None)
    h = int(getattr(model, 'height', 0))
    w = int(getattr(model, 'width', 0))
    if cells is None or h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=np.uint8)
    arr = np.fromiter((int(getattr(c, 'color', 0)) for c in cells), dtype=np.uint8, count=h * w)
    try:
        return arr.reshape(h, w)
    except Exception:
        return np.array(arr, copy=False)


essential_agent_cols = (
    'row','col','assets','num_elections_participated','personality_idx','confidence'
)

essential_area_cols = (
    'num_agents','num_cells','voter_turnout'
)


def _agent_features(model) -> np.ndarray:
    agents: List[Optional[Any]] = list(getattr(model, 'voting_agents', []) or [])
    rows: List[List[float]] = []
    for a in agents:
        if a is None:
            continue
        rows.append([
            float(getattr(a, 'row', 0)),
            float(getattr(a, 'col', 0)),
            float(getattr(a, 'assets', 0)),
            float(getattr(a, 'num_elections_participated', 0)),
            float(-1 if getattr(a, 'personality_idx', None) is None else getattr(a, 'personality_idx')),
            float(getattr(a, 'confidence', 0.0)),
        ])
    if not rows:
        return np.zeros((0, len(essential_agent_cols)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _area_features(model) -> np.ndarray:
    areas: List[Optional[Any]] = list(getattr(model, 'areas', []) or [])
    rows: List[List[float]] = []
    for ar in areas:
        if ar is None:
            continue
        rows.append([
            float(getattr(ar, 'num_agents', 0)),
            float(getattr(ar, 'num_cells', 0)),
            float(getattr(ar, 'voter_turnout', 0)),
        ])
    if not rows:
        return np.zeros((0, len(essential_area_cols)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def extract_state(model) -> Dict[str, Any]:
    agent_feats = _agent_features(model)
    area_feats = _area_features(model)
    grid = _snapshot_grid(model)
    return {
        'agent_features': agent_feats,
        'area_features': area_feats,
        'grid_tensor': grid,
    }
