# Metric Glossary (Frozen IDs and Meanings)

This glossary is the field-to-meaning dictionary for thesis analysis.
It complements:

- `docs/research/thesis_measurement_spec.md`
- `docs/research/execution_scope_freeze.md`

Metric IDs are immutable after freeze unless a documented validity bug requires correction.

## Current Contract (Implemented Truth)

### Primary time-series metrics

| Metric ID | Definition | Source | Unit | Direction | Status |
| --- | --- | --- | --- | --- | --- |
| `turnout_pct_t` | global participation rate at step `t` | `steps.turnout` | `0..100` | higher = more participation | `implemented` |
| `gini_assets_t` | inequality over agent assets at step `t` | `steps.gini_index` | `0..100` | higher = more inequality | `implemented` |
| `mean_dissatisfaction_t` | mean `agents.dissatisfaction_value` at step `t` | derived from `agents.parquet` | `0..1` | higher = worse | `implemented` |
| `gini_dissatisfaction_t` | Gini over `agents.dissatisfaction_value` at step `t` | derived from `agents.parquet` | `0..100` | higher = more inequality | `implemented` |
| `dist_to_reality_t` | eligible-weighted election quality distance at step `t` | derived from `area_steps.parquet` | `0..1` | lower = better | `implemented` |

Frozen aggregation semantics:

- `turnout_pct_t = 100 * sum_a participants(a,t) / sum_a area_num_agents(a)`; if denominator is zero, value is `0`.
- `dist_to_reality_t = sum_a dist_to_reality(a,t) * eligible_voters(a,t) / sum_a eligible_voters(a,t)`; if denominator is zero, value is `NaN`.

### Secondary descriptive metrics

| Metric ID | Definition | Source | Unit | Direction | Status |
| --- | --- | --- | --- | --- | --- |
| `mean_altruism_t` | mean altruistic-vote propensity over agents at step `t` | `steps.mean_altruism` | `0..1` | descriptive only | `implemented` |
| `diversity_first_choice_entropy_t` | normalized entropy of first-choice ballot IDs | `votes.parquet` | `0..1` | higher = more diverse | `implemented` |
| `dist_to_ref_utilitarian` | distance to utilitarian benchmark trajectory | analysis output | `0..1` | lower = closer | `implemented` |
| `dist_to_ref_nash` | distance to nash benchmark trajectory | analysis output | `0..1` | lower = closer | `implemented` |
| `dist_to_ref_egalitarian` | distance to egalitarian benchmark (`lambda=1`) | analysis output | `0..1` | lower = closer | `implemented` |
| `dist_to_ref_rawlsian` | distance to rawlsian benchmark trajectory | analysis output | `0..1` | lower = closer | `implemented` |
| `dist_to_ref_egalitarian_lam025` | egalitarian sensitivity (`lambda=0.25`) | analysis output | `0..1` | lower = closer | `implemented` |
| `dist_to_ref_egalitarian_lam400` | egalitarian sensitivity (`lambda=4.0`) | analysis output | `0..1` | lower = closer | `implemented` |

## Freeze-Target Contract (Decided, May Include Pending Items)

### Run-level thesis endpoint IDs

| Metric ID | Definition | Status |
| --- | --- | --- |
| `turnout_mean_over_time` | mean of `turnout_pct_t` over all recorded steps | `freeze-target pending` |
| `turnout_late_mean` | mean over final 20% of steps | `freeze-target pending` |
| `turnout_early_late_delta` | late mean minus early mean | `freeze-target pending` |
| `turnout_volatility` | step-change volatility endpoint | `todo pending` |
| `gini_assets_mean_over_time` | mean of `gini_assets_t` | `freeze-target pending` |
| `gini_dissatisfaction_mean_over_time` | mean of `gini_dissatisfaction_t` | `freeze-target pending` |
| `dist_to_reality_mean_over_time` | mean of `dist_to_reality_t` | `freeze-target pending` |

Rule-family status labels for inference:

- canonical family: confirmatory
- random-reference family: reference-only unless explicitly reclassified

## Freeze Status Notes

- Endpoint IDs listed under freeze-target remain fixed once promoted to required thesis inference outputs.
- Family classification remains fixed: canonical family is confirmatory, random-reference family is reference-only unless explicitly reclassified.

## Freeze Rule

After Gate B/D0 lock:

- Do not rename metric IDs.
- Do not change metric meanings without explicit decision-log entry.
- Allowed edits are limited to wording clarifications and bug-fix annotations.
