# Core Contract Drift Register (D0)

Purpose: reconcile core thesis docs with current implemented behavior and decided freeze-target behavior.

Status labels:
- `CURRENT`: implemented truth in code/output today.
- `FREEZE-TARGET`: decided contract for thesis freeze, may still be pending implementation.
- `TODO-POST-IMPLEMENTATION`: intentionally deferred item, kept explicit to avoid silent drift.

Source-of-truth files used for D0:
- `docs/research/thesis_contract.md`
- `docs/research/thesis_measurement_spec.md`
- `docs/research/metric_glossary.md`
- `docs/research/execution_scope_freeze.md`
- `docs/technical/api/output_schema_v2.md`
- `src/logging/output_schema.py`
- `src/logging/run_logger.py`
- `scripts/run_headless.py`
- `src/analysis/summary_tooling.py`
- `src/analysis/doe_runner.py`

## Drift Table

| ID | Doc statement (before D0) | Current code/output behavior | Freeze-target behavior | Status | Action |
|---|---|---|---|---|---|
| D0-01 | Contract text did not explicitly lock baseline mechanics | Current runs use `quality_target_mode=puzzle`, `participation_signal_mode=group_relative_delta_rel_party`, `altruism_mode=satisfaction` | Keep these as baseline contract for thesis trajectory | `CURRENT` | Update `thesis_contract.md` current layer with explicit baseline semantics |
| D0-02 | Contract did not include random reference rule arm | Voting rule registry now includes `random` (`rule_idx=4`) in addition to canonical rules | Keep `random` as reference arm (`rule_idx=4`), non-confirmatory by default | `CURRENT` | Code + docs aligned; keep reference-family framing frozen |
| D0-03 | Measurement spec listed volatility as if present | `summary_stats.json` currently emits means/finals (no volatility keys) | Keep volatility as explicit inference TODO until implemented | `TODO-POST-IMPLEMENTATION` | Separate "currently emitted" vs "thesis inference endpoints" in measurement spec |
| D0-04 | Glossary mixed implemented and planned terms without status tags | Core primary IDs are implemented; some endpoints are analysis-only or pending | Keep IDs immutable; add per-metric status (`implemented` / `freeze-target pending`) | `CURRENT` + `FREEZE-TARGET` | Add status column and freeze rule in glossary |
| D0-05 | Execution freeze doc had no docs-reconciliation gate | Gate A/B status exists; Gate C open | Add D0 as formal prerequisite and Gate C prereq for docs-sync closure | `FREEZE-TARGET` | Update execution scope gates and prerequisites |
| D0-06 | Output schema doc still relied on legacy static overlay `.npy` artifacts | Current writer emits `static_cell_areas.parquet` and `static_cell_agents.parquet` and references them in `static.json` | Keep schema-v2 stable and document current artifact layout | `CURRENT` | Update run layout section in `output_schema_v2.md` |
| D0-07 | Grid snapshot semantics were partly stale (`grid_0000` wording) | Current behavior: election-time snapshots `grid_001` and final always; `grid_000` only when `store_grid=true`; pad width depends on `num_steps` | Keep behavior; document clearly with pad-aware naming examples | `CURRENT` | Update grid semantics in `output_schema_v2.md` |
| D0-08 | Gate docs did not force dependency tracking per pending freeze item | Pending items existed but not bound to owners/checklists | Every pending freeze item must have explicit dependency line and owner | `FREEZE-TARGET` | Add dependency checklists in each core doc |

## D0 Dry Validation (Current Run Layout)

Validation target used: `data/simulation_output/doe_20260302_175623/design_0032/rule_approval/seed_00429/run_0`

Observed (current contract):
- Parquet artifacts present: `steps.parquet`, `area_steps.parquet`, `agents.parquet`, `votes.parquet`
- Static overlays present: `static_cell_areas.parquet`, `static_cell_agents.parquet`
- Metadata present: `meta.yaml`, `static.json`
- Grid semantics (with DOE `store_grid=false`): first+last election-time snapshots present (`grid_001.npy`, `grid_250.npy`), no `grid_000.npy`

Conclusion: run layout matches current writer/headless implementation and should be reflected in schema docs.

## D0 Sign-Off Block

- D0 contradictions resolved in docs: `READY`
- Pending freeze-target dependencies checklist complete: `READY`
- Drift register sign-off by thesis lead: `PENDING`

D0 gate is complete for implementation handoff once thesis-lead sign-off is recorded.
