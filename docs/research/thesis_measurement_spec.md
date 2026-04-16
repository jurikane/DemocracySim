# Thesis Measurement Spec

This page specifies how the thesis metrics are computed from logged artifacts
and how they are interpreted. Use it together with:

- [docs/research/thesis_contract.md](thesis_contract.md)
- [docs/research/metric_glossary.md](metric_glossary.md)

## Specification

### Data sources (logged artifacts only)

- `steps.parquet`
- `area_steps.parquet`
- `agents.parquet`
- `votes.parquet`
- `meta.yaml`
- `static.json`

### Primary time-series definitions

- `turnout_pct_t` from `steps.turnout` (0..100)
- `gini_assets_t` from `steps.gini_index` (0..100)
- `mean_dissatisfaction_t` from mean of `agents.dissatisfaction_value` by step
- `gini_dissatisfaction_t` from stepwise Gini over `agents.dissatisfaction_value` (0..100)
- `quality_distance_t` from eligible-weighted area quality distance by step

Mode-aware quality distance source:

- if `quality_target_mode = puzzle`: per-area value is `area_steps.puzzle_distance`
- if `quality_target_mode = reality`: per-area value is `area_steps.dist_to_reality`

Current weighted aggregation for `quality_distance_t`:

- numerator: `sum_a quality_distance(a,t) * eligible_voters(a,t)`
- denominator: `sum_a eligible_voters(a,t)`
- if denominator is zero: `NaN`

### Run-level summaries (`summary_stats.json`)

`global_summary` keys:

- `turnout_mean`, `turnout_final`
- `turnout_volatility`
- `gini_assets_mean`, `gini_assets_final`
- `gini_assets_volatility`
- `gini_dissatisfaction_mean`, `gini_dissatisfaction_final`
- `gini_dissatisfaction_volatility`
- `mean_dissatisfaction_mean`, `mean_dissatisfaction_final`
- `quality_distance_mean`, `quality_distance_final`
- `quality_distance_volatility`
- `diversity_entropy_mean`, `diversity_entropy_final`

Volatility definition (adjacent-step):

- step_volatility_l1(x) = mean_t |x_t - x_{t-1}| over finite adjacent pairs
- normalization:
  - turnout/gini series: divide by 100 (series are 0..100)
  - distance series (quality_distance): divide by 1
- no clamping is applied in formula layer
- if fewer than one finite adjacent pair exists: NaN

### Secondary descriptive metrics

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

Group-level descriptive diagnostics (non-confirmatory) may additionally be computed in analysis artifacts to inspect participation composition over time.

### Benchmark reference computation

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

Computation-layer assumptions:

- all `dist_to_ref_*` values are computed in analysis from logged artifacts
- no runtime reward-loop dependency on these benchmark trajectories
- optimization/tie policies are deterministic for fixed input artifacts

NaN policy:

- area-level benchmark distances are `NaN` for empty-area agent sets
- global benchmark distance is `NaN` only if the global agent set is empty
- no-vote steps still produce defined distances based on color distributions

### Consistency checks (must hold)

- `steps.turnout(t) == 100 * sum_a participants(a,t) / sum_a area_num_agents(a)` (if denominator is zero, turnout is `0`)
- `area_steps.participants(a,t) == count(votes rows for (a,t))`
- one `agents` row per `(agent_id, step)`
- no `NaN/inf` in thesis-critical emitted series (except explicitly allowed `NaN` semantics like denominator-zero `quality_distance_t`)

## Inference Specification

### Confirmatory endpoint subset

Primary confirmatory endpoints are time means:

- `turnout_mean`
- `gini_assets_mean`
- `gini_dissatisfaction_mean`
- `quality_distance_mean`

These endpoints are computed from the time-series definitions above using fixed formulas.

### Secondary reported endpoints

Additional `summary_stats.json` endpoints (final values, volatility, diversity entropy, mean dissatisfaction) are reported descriptively unless explicitly promoted in a separate analysis contract.

### Inference-family rules

Rule-family roles for reporting:

- canonical confirmatory family: `utilitarian (2)`, `borda (3)`, `schulze (4)`
- reference family: `plurality (0)`, `random (5)`
- context-only calibration arm: `approval (1)`

Family boundaries must remain explicit in results reporting.

### Multiple-testing policy rule

- We use a predefined multiplicity correction policy within each reported family.
- We do not merge canonical confirmatory and reference-family p-value pools.
- We keep endpoint formulas fixed within one reported experiment set.

### Summary-layer separation rule

- `summary_stats.json` contains the baseline run-level endpoint set.
- Additional thesis inference outputs may extend beyond this set, but must be computed from logged artifacts with fixed formulas.
