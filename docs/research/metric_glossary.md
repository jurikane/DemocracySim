# Metric Glossary

This glossary is the field-to-meaning dictionary for thesis analysis.
It complements:

- `docs/research/thesis_measurement_spec.md`

### Primary time-series metrics

| Metric ID | Definition | Source | Unit | Direction |
| --- | --- | --- | --- | --- | --- |
| `turnout_pct_t` | global participation rate at step `t` | `steps.turnout` | `0..100` | higher = more participation |
| `gini_assets_t` | inequality over agent assets at step `t` | `steps.gini_index` | `0..100` | higher = more inequality |
| `mean_dissatisfaction_t` | mean `agents.dissatisfaction_value` at step `t` | derived from `agents.parquet` | `0..1` | higher = worse |
| `gini_dissatisfaction_t` | Gini over `agents.dissatisfaction_value` at step `t` | derived from `agents.parquet` | `0..100` | higher = more inequality |
| `quality_distance_t` | eligible-weighted quality-gate distance at step `t` (mode-aware source) | derived from `area_steps.parquet` | `0..1` | lower = better |

Aggregation semantics:

- `turnout_pct_t = 100 * sum_a participants(a,t) / sum_a area_num_agents(a)`; if denominator is zero, value is `0`.
- `quality_distance_t = sum_a quality_distance(a,t) * eligible_voters(a,t) / sum_a eligible_voters(a,t)`; if denominator is zero, value is `NaN`.
- `quality_distance(a,t)` source:
  - puzzle mode: `area_steps.puzzle_distance`
  - reality mode: `area_steps.dist_to_reality`
- `dist_to_reality_t` remains a secondary grid-alignment diagnostic (not the primary gate metric in puzzle mode).

### Secondary descriptive metrics

| Metric ID | Definition | Source | Unit | Direction |
| --- | --- | --- | --- | --- |
| `mean_altruism_t` | mean altruistic-vote propensity over agents at step `t` | `steps.mean_altruism` | `0..1` | descriptive only |
| `diversity_first_choice_entropy_t` | normalized entropy of first-choice ballot IDs | `votes.parquet` | `0..1` | higher = more diverse |
| `dist_to_ref_utilitarian` | distance to utilitarian benchmark trajectory | analysis output | `0..1` | lower = closer |
| `dist_to_ref_nash` | distance to nash benchmark trajectory | analysis output | `0..1` | lower = closer |
| `dist_to_ref_egalitarian` | distance to egalitarian benchmark (`lambda=1`) | analysis output | `0..1` | lower = closer |
| `dist_to_ref_rawlsian` | distance to rawlsian benchmark trajectory | analysis output | `0..1` | lower = closer |
| `dist_to_ref_egalitarian_lam025` | egalitarian sensitivity (`lambda=0.25`) | analysis output | `0..1` | lower = closer |
| `dist_to_ref_egalitarian_lam400` | egalitarian sensitivity (`lambda=4.0`) | analysis output | `0..1` | lower = closer |

### Run-level thesis endpoint IDs

| Metric ID | Definition |
| --- | --- |
| `turnout_mean` | mean of `turnout_pct_t` over all recorded steps |
| `turnout_final` | final observed `turnout_pct_t` value |
| `turnout_volatility` | adjacent-step L1 volatility endpoint |
| `gini_assets_mean` | mean of `gini_assets_t` |
| `gini_assets_final` | final observed `gini_assets_t` value |
| `gini_assets_volatility` | adjacent-step L1 volatility endpoint |
| `gini_dissatisfaction_mean` | mean of `gini_dissatisfaction_t` |
| `gini_dissatisfaction_final` | final observed `gini_dissatisfaction_t` value |
| `gini_dissatisfaction_volatility` | adjacent-step L1 volatility endpoint |
| `mean_dissatisfaction_mean` | mean of `mean_dissatisfaction_t` |
| `mean_dissatisfaction_final` | final observed `mean_dissatisfaction_t` value |
| `quality_distance_mean` | mean of `quality_distance_t` |
| `quality_distance_final` | final observed `quality_distance_t` value |
| `quality_distance_volatility` | adjacent-step L1 volatility endpoint |
| `diversity_entropy_mean` | mean of diversity entropy over time |
| `diversity_entropy_final` | final observed diversity entropy |

Rule-family labels used in inference:

- canonical family: confirmatory
- random-reference family: reference-only unless explicitly reclassified
