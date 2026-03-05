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
- `base_seed=0` is valid and treated as an explicit deterministic seed (not as missing).
- `run_seed` is written to `meta.yaml` and included in all parquet tables.
- Same config + same `run_id` + same `base_seed` must reproduce identical outputs.

### `num_steps`, `runs`

- `num_steps` controls recorded steps (`1..num_steps`) in parquet tables.
- `runs` controls number of run directories (`run_0..run_{runs-1}`) in a batch.

### `store_grid`, `grid_interval`, filename/indexing

- First/last election-time snapshots are always written: `step=1` and `step=num_steps`.
- `store_grid=True`: initial pre-election snapshot `grid_000..0.npy` is also written as step 0.
- `store_grid=True`: additional per-step snapshots are written according to interval:
  `step=1, 1+grid_interval, ...` (plus always-final `step=num_steps`).
- `store_grid=False`: interval snapshots and step 0 are disabled, but first/last election-time snapshots remain enabled.
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

### Config parsing strictness

- Config models are strict (`extra="forbid"`).
- Unknown/stale keys fail at load time (no silent dropping).
- Deprecated legacy keys (for example `common_assets`) are rejected.

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
- Freeze run outputs with `config_used.yaml` + commit hash/version string for traceability.

## Summary Tooling (G4, Batch 1 + 2)

Run-level summaries are generated from logged artifacts only:

```bash
python -m scripts.generate_summary --run-dir <path_to_run_dir>
```

Canonical visualization planning/inventory (thesis-first):

- `docs/research/visualization_insight_inventory.md`

Mode selection:

- `--mode full` (default): compute all benchmark families (`utilitarian`, `nash`, `rawlsian`, `egalitarian` + sensitivity)
- `--mode fast`: compute only `utilitarian` and `nash`; expensive benchmark families are skipped (distance columns kept as `NaN`)

Reference-cache control:

- default uses cache files in `analysis/` (`reference_cache_full.json` / `reference_cache_fast.json`)
- pass `--no-cache` to force recomputation

By default, generated summary PDFs are auto-opened (best-effort) in reverse launch order
so stacked windows appear in reading order. To disable auto-opening:

```bash
python -m scripts.generate_summary --run-dir <path_to_run_dir> --closed
```

Current outputs:

- `analysis/summary_global_series.csv`
- `analysis/summary_area_series.csv`
- `analysis/summary_area_group_series.csv`
- `analysis/summary_stats.json`
- `analysis/global_summary_<rule>_seed<seed>.pdf` (combined document)
- `analysis/area_<id>.pdf` (one detail report per area)

Planned next G4 outputs:

- `analysis/areas_overview.pdf`

`global_summary_<rule>_seed<seed>.pdf` currently includes, in this order:

1. fixed reference optima + global color curves + grid snapshots
2. static overview page(s)
3. remaining global metric/diagnostic pages

- core global thesis metrics
- `dist_to_reality`, `dist_to_ref_*`, diversity diagnostics

`area_<id>.pdf` currently includes:

1. turnout + participants + gini assets
2. area color curves + `dist_to_reality` (+ elected-ordering background) + `dist_to_ref_*`
3. group diagnostics (participants/eligible/residents + turnout by group)
4. group means (assets, dissatisfaction)
5. group composition diagnostics

Benchmark-reference note:

- `dist_to_ref_*` used in summary artifacts are computed post-run in analysis from
  logged color distributions + static preferences (not from simulation-time reward logic).
- Current benchmark families in analysis outputs:
  - utilitarian (`L2^2` mean reference)
  - nash (`KL` geometric-mean reference)
  - egalitarian (`mean + lambda * Gini`, with `lambda` sensitivity)
  - rawlsian (minimax `L2^2`)

## DOE Screening Runner (Phase 1)

Run phase-1 design-of-experiments screening (approval primary + utilitarian robustness):

```bash
python -m scripts.run_doe --points 48 --seeds 101,202,303
```

Stratified seed mode (spread-out initial conditions from a candidate pool):

```bash
python -m scripts.run_doe --points 48 --seed-mode stratified --seed-target 3 --seed-candidate-count 40
```

Dry-run safety check (writes manifests only, no simulation execution):

```bash
python -m scripts.run_doe --points 3 --seeds 101,202 --no-robustness --dry-run
```

Notes:

- Applies the confirmed phase-1 DOE profile (frozen structure + tunable ranges).
- Profile can be switched with `--doe-profile`:
  - `phase1` (default, tunes `altruism_static`, keeps `altruism_learning=false`)
  - `phase2_altruism_learning` (enables `altruism_learning=true`, tunes `altruism_alpha` + `altruism_init`)
  - `phase2_altruism_probe` (small side-DOE: fixes non-altruism knobs and tunes `altruism_alpha`, `altruism_init`, `satisfaction_baseline_alpha`, `known_cells`)
- Default config is `configs/doe.yaml` (override with `--config` if needed).
- Default output root: `data/simulation_output/doe_<timestamp>/`.
- Robustness cadence can be reduced with `--robust-every N` or disabled with `--no-robustness`.
- Batch safety: pass `--continue-on-error` to keep running after per-run failures.
- Seed selection metadata is written to `doe_seed_selection.json`.
- Planned run rows are written to `doe_run_manifest.csv` (design/seed/rule/output/params-hash).

Example phase-2 run:

```bash
python -m scripts.run_doe --doe-profile phase2_altruism_learning --points 48 --seeds 101,202,303
```

Example phase-2 probe run:

```bash
python -m scripts.run_doe --doe-profile phase2_altruism_probe --points 36 --seeds 101,202,303 --no-robustness
```

## DOE Scoring Pipeline

Score DOE outputs with hard gates + weighted ranking:

```bash
python -m scripts.score_doe --doe-root data/simulation_output/doe_<timestamp> --objective-config configs/doe_selection_objective_v1.json
```

`scripts.score_doe` defaults to `configs/doe_selection_objective_v1.json`; pass `--objective-config` only to switch objective contract explicitly.

Default behavior is **strict completeness filtering**:

- only designs with full expected primary seed coverage are ranked
- if robustness is enabled with `robust_every=1`, only designs with full matched robust pairs are ranked

Disable this (exploratory only):

```bash
python -m scripts.score_doe --doe-root data/simulation_output/doe_<timestamp> --allow-incomplete-designs
```

Output artifacts (written to DOE root by default):

- `doe_run_features.csv` (per-run extracted metrics + gate flags)
- `doe_design_scores.csv` (aggregated per-design ranking table)
- `doe_selection_spec.json` (authoritative selection stage spec: objective path, thresholds, weights, score formula)
- `doe_top_designs.json` (top-5 shortlist)

## DOE HIL Review Helper

Use queued representative runs for top/mid/bottom visual validation:

```bash
python -m tools.doe.doe_hil_review --print-commands
```

Populate AI interpretation text for selected queue rows:

```bash
python -m tools.doe.doe_hil_review --populate-ai
```

Execute fast summary generation for selected queue rows:

```bash
python -m tools.doe.doe_hil_review --bucket top --limit 5 --run-fast
```

Recommended HIL queue columns:

- `ai_interpretation` (auto-filled from gates + score components via `--populate-ai`)
- `human_feedback` (free-text assessment after reading the summary PDF)
- `human_verdict` (`top_like|mid_like|bottom_like`)
- `adjustment_hint` (optional metric/gate update hint)

Current hard gates:

- no-pathological collapse (`max_all_abstain_stretch <= 10`)
- no-early-lock-in (`winner_changes_post_burnin >= 3`)
- not-too-chaotic (`winner_changes_post_burnin <= 80`)
- short-window divergence present (`roll3_group_turnout_range_max >= 0.3`)
- medium-window divergence present (`roll20_group_turnout_range_max >= 0.1`)
- winner-order diversity (`winner_entropy_norm >= 0.25`)
- reality-distance activity (`dist_nonzero_share >= 0.05`)
- competitive group dynamics present (`competitive_step_share >= 0.05`)
- turnout in usable band (`20 <= mean_turnout <= 90`)
- signal present (`turnout_std >= 0.5` OR `gini_std >= 1.0` OR `dist_std >= 0.02`)

Soft (scored) calibration pressures:

- cross-group divergence pressure is tracked in score via:
  - `group_turnout_range_mean`
  - `roll3_group_turnout_range_mean`
  - `roll3_group_turnout_range_max`
  - `roll20_group_turnout_range_mean`
  - `roll20_group_turnout_range_max`
  - `group_turnout_residual_abs_mean`
  - `group_participant_abstainer_delta_rel_gap_abs`
- winner/reality diversity pressure is tracked in score via:
  - `winner_entropy_norm`
  - `dist_nonzero_share`
  - `competitive_step_share`
- lock-in pressure is still scored via `winner_changes_post_burnin` (in addition to hard minimum + hard maximum gates).

Current aggregate score (refactored, non-multiplicative):

- `quality_bundle = 0.45*quality_mean + 0.35*discriminability + 0.20*seed_robustness`
- `score_total = 0.60*pass_rate + 0.40*quality_bundle`
- `quality_mean` currently includes participant-vs-abstainer separation via `participant_abstainer_delta_rel_gap_abs`
- `quality_mean` also includes group divergence components (`group_turnout_range_mean`, `group_turnout_residual_abs_mean`, `group_participant_abstainer_delta_rel_gap_abs`)
- `quality_mean` also includes rolling-window divergence (`roll3_group_turnout_range_mean`, `roll3_group_turnout_range_max`, `roll20_group_turnout_range_mean`, `roll20_group_turnout_range_max`)
- if no robustness rule data is present, discriminability is auto-disabled and effective quality/robustness weights are renormalized (recorded in `doe_selection_spec.json`)

Additional extracted (window-based) diagnostics now available in `doe_run_features.csv`:

- `roll10_dist_std_mean` (average 10-step rolling std of `dist_to_reality`)
- `roll10_winner_change_rate` (average within-window winner-change rate over 10-step windows)
- `roll10_turnout_slope_abs_mean` (average absolute linear slope of turnout over 10-step windows)
- `roll10_group_sync_index` (average pairwise group-turnout correlation over 10-step windows)
- `lag1_group_signal_turnout_response_corr` (correlation of group mean signal at `t` with group turnout change `t -> t+1`)

Important semantic note:

- `burn_in_steps` in DOE scoring is only an **analysis warm-up exclusion window** for lock-in/chaos metrics.
  It does **not** change simulation dynamics, does not reset state, and does not implement burn-in logic in the model.
- default is `0` (no exclusion); pass `--burn-in-steps N` only if you explicitly want a warm-up window.

## DOE Inference Stage

Build DOE inference artifacts from existing DOE outputs:

```bash
python -m tools.research.doe_inference --doe-root data/simulation_output/doe_<timestamp> --bootstrap-reps 500 --random-seed 11
```

Outputs:

- `doe_inference_spec.json` (authoritative inference stage spec + artifact pointers)
- `doe_seed_fixed_effects.csv` (seed-adjusted fixed-effects coefficients + bootstrap CIs)
- `doe_nonlinear_importance.csv` (quantile-bin eta-squared nonlinear importance)
- `doe_interaction_maps.csv` (2D interaction cell means for top knobs)
- `doe_bootstrap_design_ci.csv` (design-level bootstrap confidence intervals)
- `doe_pareto_designs.csv` (Pareto non-dominated designs on core objectives)

Notes:

- the seed-adjusted model uses **seed fixed effects OLS** (with bootstrap CIs)
  as the runtime-safe replacement for mixed-effects when `statsmodels` is unavailable.
- this is the dedicated inference stage; no legacy correlation/elite-range artifacts are emitted.
