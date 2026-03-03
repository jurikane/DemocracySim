# Thesis Measurement Spec (Core Contract)

This document defines how thesis metrics are computed and interpreted.
It is coupled with:

- `docs/research/thesis_contract.md`
- `docs/research/metric_glossary.md`
- `docs/research/execution_scope_freeze.md`

## Current Contract (Implemented Truth)

### Data sources (logged artifacts only)

- `steps.parquet`
- `area_steps.parquet`
- `agents.parquet`
- `votes.parquet`
- `meta.yaml`
- `static.json`

### Current primary time-series definitions

- `turnout_pct_t` from `steps.turnout` (0..100)
- `gini_assets_t` from `steps.gini_index` (0..100)
- `mean_dissatisfaction_t` from mean of `agents.dissatisfaction_value` by step
- `gini_dissatisfaction_t` from stepwise Gini over `agents.dissatisfaction_value` (0..100)
- `dist_to_reality_t` from eligible-weighted `area_steps.dist_to_reality` by step

Current weighted aggregation for `dist_to_reality_t`:

- numerator: `sum_a dist_to_reality(a,t) * eligible_voters(a,t)`
- denominator: `sum_a eligible_voters(a,t)`
- if denominator is zero: `NaN`

### Current emitted run-level summaries (`summary_stats.json`)

Current `global_summary` keys emitted by `summary_tooling`:

- `turnout_mean`, `turnout_final`
- `gini_assets_mean`, `gini_assets_final`
- `gini_dissatisfaction_mean`, `gini_dissatisfaction_final`
- `mean_dissatisfaction_mean`, `mean_dissatisfaction_final`
- `dist_to_reality_mean`, `dist_to_reality_final`
- `diversity_entropy_mean`, `diversity_entropy_final`

Not emitted in current sidecar summary:

- `turnout_volatility`
- analogous volatility endpoints for other primary metrics

### Current secondary descriptive metrics

- `mean_altruism_t` from `steps.mean_altruism` (mechanism diagnostic; non-confirmatory)
- `diversity_first_choice_entropy_t` from `votes.rank_1_option_id`
- `dist_to_ref_*` benchmark trajectories (analysis artifacts; not runtime schema columns)
  - `dist_to_ref_utilitarian`
  - `dist_to_ref_nash`
  - `dist_to_ref_egalitarian`
  - `dist_to_ref_rawlsian`
  - `dist_to_ref_egalitarian_lam025`
  - `dist_to_ref_egalitarian_lam400`

These are descriptive benchmark comparisons, not normative optimality claims.

Group-level descriptive diagnostics (non-confirmatory) may additionally be computed in analysis artifacts to inspect majority/minority participation composition over time.

### Current benchmark reference computation contract

Reference families currently used in analysis:

- utilitarian reference
- nash reference
- egalitarian reference (`lambda=1.0`) with sensitivity variants (`lambda=0.25`, `lambda=4.0`)
- rawlsian reference

Analysis output columns (time-indexed):

- `dist_to_ref_utilitarian`
- `dist_to_ref_nash`
- `dist_to_ref_egalitarian`
- `dist_to_ref_rawlsian`
- `dist_to_ref_egalitarian_lam025`
- `dist_to_ref_egalitarian_lam400`

Computation-layer freeze:

- all `dist_to_ref_*` values are computed in analysis from logged artifacts
- no runtime reward-loop dependency on these benchmark trajectories
- optimization/tie policies are deterministic for fixed input artifacts

NaN policy:

- area-level benchmark distances are `NaN` for empty-area agent sets
- global benchmark distance is `NaN` only if the global agent set is empty
- no-vote steps still produce defined distances based on color distributions

### Current consistency checks (must hold)

- `steps.turnout(t) == 100 * sum_a participants(a,t) / sum_a area_num_agents(a)` (if denominator is zero, turnout is `0`)
- `area_steps.participants(a,t) == count(votes rows for (a,t))`
- one `agents` row per `(agent_id, step)`
- no `NaN/inf` in thesis-critical emitted series (except explicitly allowed `NaN` semantics like denominator-zero `dist_to_reality_t`)

## Freeze-Target Contract (Decided, May Include Pending Items)

### Thesis inference endpoints (design contract)

For each primary time series and each run, use fixed estimands:

- `mean_over_time`
- `late_mean` (last 20% of steps)
- `early_late_delta = late_mean - early_mean` (first 20% vs last 20%)
- `volatility` (step-change instability metric)

Status in freeze-target:

- `mean_over_time`, `late_mean`, `early_late_delta`: required for thesis inference layer.
- `volatility`: allowed but currently pending implementation in sidecar outputs.

### Inference-family guardrails

- Canonical rule-family tests and reference-family tests must remain separated in reporting.
- Multiple-testing correction policy is frozen before first full final-run readout.
- No endpoint formula changes after freeze lock.

### Summary-layer separation rule

- Sidecar summary (`summary_stats.json`) describes currently emitted implementation outputs.
- Thesis inference outputs may extend beyond sidecar keys, but must be computed from logged artifacts with fixed formulas.

## Implementation Status Notes

- Sidecar summary keys are intentionally limited to currently emitted implementation outputs.
- Freeze-target inference endpoints are fixed at the formula level and are computed in the thesis inference layer from logged artifacts.
- Volatility endpoints remain optional and are promoted only if explicitly required before final runs.
- Final endpoint and multiplicity lock-in is recorded in internal freeze notes before final execution.
