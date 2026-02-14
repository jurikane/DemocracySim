# Run Control & Output

This document is the single reference for **run control and output pipeline semantics**:
how runs are seeded/executed, how step/grid indexing works, and which output
contracts are enforced for thesis analysis.

## Where It Happens (Runtime Path)

Headless execution and output writing:

1. `scripts/run_headless.py::batch_run`
   - resolves output base directory
   - stores `config_used.yaml` at timestamp root
   - executes `run_once(run_id, ...)` for `run_0..run_{runs-1}`
2. `scripts/run_headless.py::run_once`
   - derives `run_seed = base_seed + run_id`
   - creates model with `model.seed = run_seed`
   - writes schema-v2 metadata/static artifacts
   - executes `num_steps` and logs parquet + optional grids
3. `src/logging/run_logger.py::RunLoggerV2`
   - writes `meta.yaml`, `static.json`, parquet tables, and grid snapshots
   - enforces schema-v2 validators before writing

## Semantics (Authoritative)

### `seed` / `base_seed` / `run_seed`

- Per-run RNG seed is deterministic: `run_seed = base_seed + run_id`.
- `run_seed` is written to `meta.yaml` and included in all parquet tables.
- Same config + same `run_id` + same `base_seed` must reproduce identical outputs.

### `num_steps`, `runs`, `processes`

- `num_steps` controls recorded steps (`1..num_steps`) in parquet tables.
- `runs` controls number of run directories (`run_0..run_{runs-1}`) in a batch.
- `processes` is currently metadata-only (batch execution is sequential).

### `store_grid`, `grid_interval`, filename/indexing

- `store_grid=False`: no `grids/` directory is written.
- `store_grid=True`: initial election-time snapshot `grid_000..0.npy` is written as step 0.
- Per-step grid snapshots are written for recorded steps according to interval:
  `step=1, 1+grid_interval, ...`.
- Replay behavior with sparse grids (`grid_interval > 1`):
  when `grid_t` is missing, replay carries forward the latest available snapshot
  at step `<= t`.
- Replay UI shows both the replay step and the grid-source step at the top
  (`Replay Step t | Grid Step k`) so carry-forward states are visible.
- Filename padding width is `len(str(num_steps))`; pattern is exposed in `static.json`.

### Output location (`output.directory`)

- Default output base: `<project_root>/data/simulation_output`.
- Absolute paths are used as-is.
- Relative paths are resolved relative to project root.
- `~` and environment variables are expanded.

### Schema-v2 fail-loud guarantees

- Missing required pre-mutation area snapshot fields fail loudly (runtime error).
- Missing/invalid vote estimate distributions are logged as `NaN` (not silent zeros).

## Why This Matters for Thesis Validity

- Reproducibility and provenance are central for defensible rule comparisons.
- Incorrect step/grid indexing can invalidate replay-based interpretation.
- Silent logging fallbacks can create “plausible but wrong” analysis inputs.

## Test Coverage (What Is Locked By Pytests)

- `tests/test_seed_run_control_contract.py`
- `tests/test_base_seed_contract.py`
- `tests/test_headless_determinism.py`
- `tests/test_num_steps_contract.py`
- `tests/test_runs_processes_contract.py`
- `tests/test_grid_storage_contract.py`
- `tests/test_output_paths.py`
- `tests/test_output_pipeline_contract.py`
- `tests/test_output_schema_v2_contract.py`

Interaction tests in this section:

- `tests/test_runs_processes_contract.py::test_h_knobs_interaction_runs_steps_and_grid_interval_in_batch_mode`
  - jointly checks `runs`, `num_steps`, `store_grid`, `grid_interval`, and run-seed progression

## Recommended Thesis Run Policy

- Fix `base_seed`, `num_steps`, `store_grid`, and `grid_interval` before final experiment batches.
- Keep `processes` fixed (or document it explicitly) even though execution is currently sequential.
- Freeze run outputs with `config_used.yaml` + commit hash/version string for traceability.
