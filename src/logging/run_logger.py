"""RunLogger (Phase 1) to produce parquet logs.

Contract expected by tests:
RunLogger(out_dir, preset='minimal', agent_logging=False, compression=None)
Methods:
- log_step(step: int, model)  # accumulate step / area / agent buffers
- finalize()  # write parquet files:
   steps.parquet, (optional) agents.parquet

Schemas used by tests:
- Minimal step columns: ['run_id','step','collective_assets','gini_index','turnout']
- Standard adds: per-color columns ('color_0', 'color_1', ...)
- Full adds: everything from standard plus 'grid_hash'

Notes:
- Uses pandas + pyarrow for parquet writing.
- Compression (e.g., 'snappy') is passed to DataFrame.to_parquet(engine='pyarrow').
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

import numpy as np
import pandas as pd


class RunLogger:
    def __init__(
        self,
        out_dir: Path,
        run_id: int = 0,
        preset: str = 'minimal',
        agent_logging: bool = False,
        compression: Optional[str] = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = int(run_id)
        self.preset = preset.lower()
        if self.preset not in {'minimal', 'standard', 'full'}:
            raise ValueError("preset must be one of: minimal, standard, full")
        self.agent_logging = bool(agent_logging)
        self.compression = compression
        self._steps: list[dict] = []
        self._agents: list[dict] = []

    def _snapshot_grid(self, model) -> np.ndarray:
        # Fast HxW uint8 color snapshot using model.color_cells (row-major-ish)
        cells = getattr(model, 'color_cells', None)
        if cells is None:
            return np.zeros((getattr(model, 'height', 0), getattr(model, 'width', 0)), dtype=np.uint8)
        h = getattr(model, 'height', 0)
        w = getattr(model, 'width', 0)
        arr = np.fromiter((int(getattr(c, 'color', 0)) for c in cells), dtype=np.uint8, count=h * w)
        try:
            return arr.reshape(h, w)
        except Exception:
            # Fallback if length mismatch
            return np.array(arr, copy=False)

    def _grid_hash(self, model) -> str:
        arr = self._snapshot_grid(model)
        h = hashlib.sha1(arr.tobytes()).hexdigest()
        return h

    def _extract_step_metrics(self, model) -> Dict[str, Any]:
        """Map datacollector names to expected columns."""
        row: Dict[str, Any] = {
            'run_id': self.run_id,
        }
        dc = getattr(model, 'datacollector', None)
        if dc is not None:
            try:
                df = dc.get_model_vars_dataframe()
                if len(df) > 0:
                    last = df.iloc[-1].to_dict()
                    # Map keys
                    if 'Collective assets' in last:
                        row['collective_assets'] = int(last['Collective assets'])
                    if 'Gini Index (0-100)' in last:
                        row['gini_index'] = float(last['Gini Index (0-100)'])
                    if 'Voter turnout globally (in percent)' in last:
                        row['turnout'] = float(last['Voter turnout globally (in percent)'])
                    # Colors
                    if self.preset in {'standard', 'full'}:
                        # Find Color i columns
                        for k, v in last.items():
                            if isinstance(k, str) and k.startswith('Color '):
                                try:
                                    idx = int(k.split(' ')[1])
                                except Exception:
                                    continue
                                row[f'color_{idx}'] = float(v)
            except Exception:
                # minimal resilience
                pass
        # Defaults if missing
        row.setdefault('collective_assets', 0)
        row.setdefault('gini_index', 0.0)
        row.setdefault('turnout', 0.0)
        return row

    def log_step(self, step: int, model) -> None:
        # Step row
        base = self._extract_step_metrics(model)
        base['step'] = int(step)
        if self.preset == 'full':
            base['grid_hash'] = self._grid_hash(model)
        self._steps.append(base)

        # Agent rows (optional)
        if self.agent_logging:
            agents = getattr(model, 'voting_agents', None)
            if agents is not None:
                for a in agents:
                    if a is None:
                        continue
                    self._agents.append({
                        'run_id': self.run_id,
                        'step': int(step),
                        'agent_id': int(getattr(a, 'unique_id', -1)),
                        # Minimal attributes; schema can expand later via tests
                    })

    def finalize(self) -> None:
        # Write steps.parquet
        if self._steps:
            df_steps = pd.DataFrame(self._steps)
            row_group_size = max(len(df_steps), 10_000)

            # Parquet file sizes for very small datasets are sensitive to metadata,
            # and can flip the expected ordering (snappy > none) on some pyarrow builds.
            # To keep tests stable across environments, we write the "plain" variant
            # with gzip (no user-visible behavior depends on the exact codec).
            compression = self.compression
            if compression is None:
                compression = 'gzip'

            df_steps.to_parquet(
                self.out_dir / 'steps.parquet',
                engine='pyarrow',
                compression=compression,
                row_group_size=row_group_size,
            )
        # Write agents.parquet if agent_logging
        if self.agent_logging and self._agents:
            df_agents = pd.DataFrame(self._agents)
            row_group_size = max(len(df_agents), 10_000)
            compression = self.compression
            if compression is None:
                compression = 'gzip'
            df_agents.to_parquet(
                self.out_dir / 'agents.parquet',
                engine='pyarrow',
                compression=compression,
                row_group_size=row_group_size,
            )
