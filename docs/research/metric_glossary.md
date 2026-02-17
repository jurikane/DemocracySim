# Metric Glossary (Frozen Names & Meanings)

This is the field-to-meaning dictionary for thesis analysis.
It complements:

- `docs/research/thesis_measurement_spec.md` (formulas and analysis rules)
- `docs/research/execution_scope_freeze.md` (scope and change control)

## Primary Metrics

| Metric ID | Definition | Source | Unit | Direction |
|---|---|---|---|---|
| `turnout_pct_t` | global participation rate at step `t` | `steps.turnout` | percent (`0..100`) | higher = more participation |
| `gini_assets_t` | inequality over agent assets at step `t` | `steps.gini_index` | `0..100` | higher = more inequality |
| `mean_dissatisfaction_t` | mean of `agents.dissatisfaction_value` at step `t` | derived from `agents.parquet` | `0..1` | higher = worse |
| `gini_dissatisfaction_t` | Gini over `agents.dissatisfaction_value` at step `t` | derived from `agents.parquet` | `0..100` | higher = more inequality |
| `dist_to_reality_t` | area-level election quality aggregated at step `t` | derived from `area_steps.parquet` | `0..1` | lower = better |

`turnout_pct_t` aggregation rule (frozen):

- population-based aggregation across areas:
  - `100 * sum_a participants(a,t) / sum_a area_num_agents(a)`
- if denominator is zero, define value as `0`.

`dist_to_reality_t` aggregation rule (frozen):

- eligible-weighted mean across areas:
  - `sum_a dist_to_reality(a,t) * eligible_voters(a,t) / sum_a eligible_voters(a,t)`
- if denominator is zero, define value as `NaN` (or skip step in summaries).

## Secondary Descriptive Metrics

| Metric ID | Definition | Source | Unit | Direction |
|---|---|---|---|---|
| `diversity_first_choice_entropy_t` | normalized entropy of `rank_1_option_id` distribution at step `t` | `votes.parquet` | `[0,1]` | higher = more diverse |
| `dist_to_ref_utilitarian` | distance to utilitarian benchmark reference | analysis output | `0..1` | lower = closer |
| `dist_to_ref_nash` | distance to nash benchmark reference | analysis output | `0..1` | lower = closer |
| `dist_to_ref_egalitarian` | distance to egalitarian benchmark reference (`lambda=1`) | analysis output | `0..1` | lower = closer |
| `dist_to_ref_rawlsian` | distance to rawlsian benchmark reference | analysis output | `0..1` | lower = closer |
| `dist_to_ref_egalitarian_lam025` | distance to egalitarian sensitivity reference (`lambda=0.25`) | analysis output | `0..1` | lower = closer |
| `dist_to_ref_egalitarian_lam400` | distance to egalitarian sensitivity reference (`lambda=4.0`) | analysis output | `0..1` | lower = closer |

## Naming Convention

- Use **dissatisfaction** consistently in thesis text and code/logs (`dissatisfaction_value`).

## Freeze Rule

After schema freeze (Gate B), metric IDs and meanings in this glossary are immutable.
Only bug-fix clarifications are allowed, with explicit entry in `docs/technical/decision_log.md`.
