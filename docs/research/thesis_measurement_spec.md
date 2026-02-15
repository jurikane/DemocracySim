# Thesis Measurement Spec (Frozen Before Final Runs)

This document defines the operational metrics used for final thesis analysis.
It aligns the thesis framing with the current simulation semantics and logging pipeline.
Execution-level scope and triage are frozen in:
`docs/research/execution_scope_freeze.md`.

## Data Sources

- `steps.parquet`
- `area_steps.parquet`
- `agents.parquet`
- `votes.parquet`
- `meta.yaml` / `static.json` (run metadata and schema metadata)

All analysis is derived from logged artifacts only (no live model-state dependencies).

## Core Variables

- `t`: recorded step index (1-based)
- `i`: agent index
- `a`: area index

### Participation

- `P(t) = steps.turnout[t]`
- Unit: percent (`0..100`)

### Inequality: Resource Dimension

- Agent resource state: `A_i(t) = agents.assets` at step `t`
- Inequality series: `I_A(t) = steps.gini_index[t]`
- Unit: percent-like Gini scale (`0..100`)

Interpretation: inequality in simulation participation capacity (resource / motivation to take effort), not literal income inequality.
Note: agents may be initialized equally, but inequality is evaluated dynamically over time.

### Inequality: Experiential Dimension

- Agent dissatisfaction: `D_i(t) = agents.satisfaction_value` at step `t`
- `D_i(t)` is a normalized distribution-distance style quantity (`0..1` in current implementation)
- Define:
  - `L_D(t) = mean_i D_i(t)` (mean dissatisfaction level)
  - `I_D(t) = Gini_0_100({D_i(t)})` (dissatisfaction inequality)

Interpretation: heterogeneity in experienced preference-mismatch.
Terminology convention in thesis text: use **dissatisfaction** (the current field name in code/logs remains `satisfaction_value`).

## Summary Statistics Per Run

For each run, compute:

- `turnout_mean`, `turnout_final`, `turnout_volatility`
- `gini_assets_mean`, `gini_assets_final`
- `mean_dissatisfaction_mean`, `mean_dissatisfaction_final`
- `gini_dissatisfaction_mean`, `gini_dissatisfaction_final`
- `dist_to_reality_mean`, `dist_to_reality_final` (from `area_steps`)

Recommended volatility definition:

- standard deviation over steps of the corresponding series.

## Cross-Table Consistency Checks (Must Hold)

- `steps.turnout(t) == 100 * sum_a participants(a,t) / sum_a eligible_voters(a,t)` (if denominator is 0, turnout is defined as 0)
- `area_steps.participants(a,t) == count(votes rows for (a,t))`
- One `agents` row per `(agent_id, step)`
- All probability-vector columns sum to 1 within numeric tolerance
- No `NaN/inf` in thesis-critical series

## Secondary Descriptive Metrics (Included)

These are included as descriptive diagnostics and must not be interpreted as
normative welfare-optimality claims.

### Diversity of Shared Opinions

Purpose:

- characterize whether participating ballots are convergent or dispersed at step `t`.

Operational candidates (implementation may use one or both):

- entropy of first-choice option distribution among participants
- mean pairwise ballot-distance among participants (from vote-level signals)

### Distance to Reference Optima

Purpose:

- track how far realized collective outcomes are from fixed reference outcomes
  computed from static preference information.

Reference families:

- utilitarian reference
- egalitarian reference
- rawlsian reference

Interpretation rule:

- these are benchmark trajectories for comparison, not claims about “true”
  democratic optimality.

## Experimental Freeze Rules

Before final experiment execution:

- Fix independent variable plan: vary only the voting rule.
- Fix satisfaction mode and learning knobs.
- Fix overlap mode (baseline choice) and topology settings.
- Freeze config files + commit hash used for final runs.
- Do not change metric formulas after observing final rule-comparison results.
- Treat secondary descriptive metrics as non-normative diagnostics.
