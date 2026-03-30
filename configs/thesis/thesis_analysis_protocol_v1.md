# Thesis Analysis Protocol v1

This document defines the locked analysis contract for the final thesis runs and
their reporting outputs.

## 1. Scope and authority

This protocol fixes the analysis contract for the final thesis evaluation:

- endpoint set
- hypothesis families
- test procedures
- multiplicity rules
- figure and table inventory

## 2. Rule families

- canonical confirmatory family:
  - `utilitarian (rule_idx=2)`
  - `borda (rule_idx=3)`
  - `schulze (rule_idx=4)`
- reference family:
  - `plurality (rule_idx=0)`
  - `random (rule_idx=5)`
- context-only calibration arm:
  - `approval (rule_idx=1)`

## 3. Final run matrix

- main inferential matrix: `5 rules x 200 matched seeds = 1000 runs`
- approval context matrix: `1 rule x 50 seeds = 50 runs`
- total planned runs: `1050`
- steps per run: `250`
- seed policy:
  - `S_main` = 200 matched seeds
  - `S_approval` = first 50 seeds from `S_main`

## 4. Endpoint contract

### 4.1 Confirmatory endpoints

Time means only:

- `turnout_mean`
- `gini_assets_mean`
- `gini_dissatisfaction_mean`
- `quality_distance_mean`

`quality_distance_mean` is mode-aware:

- `quality_target_mode=puzzle` -> `area_steps.puzzle_distance`
- `quality_target_mode=reality` -> `area_steps.dist_to_reality`

### 4.2 Secondary reported endpoints

- finals and volatility endpoints from `summary_stats.json`
- diversity entropy endpoints
- mean dissatisfaction endpoints
- collective assets endpoints

These are descriptive unless explicitly promoted in a later protocol revision.

Interpretation rules:

- `gini_dissatisfaction_mean` is not interpreted on its own as a welfare or
  quality result.
- whenever dissatisfaction inequality is discussed substantively, the
  corresponding descriptive dissatisfaction-level context should also be shown
  in the relevant figure, table, or local discussion.
- `gini_assets_mean` is a distributional endpoint and not a direct asset-level
  result.
- whenever asset inequality is interpreted substantively, the corresponding
  descriptive collective-asset context should also be shown where it materially
  affects interpretation.

## 5. Hypothesis families

### 5.1 Canonical confirmatory hypotheses

- H1: voting rule affects `turnout_mean`
- H2: voting rule affects `gini_assets_mean`
- H3: voting rule affects `gini_dissatisfaction_mean`
- H4: voting rule affects `quality_distance_mean`

Contrasts:

- utilitarian vs borda
- utilitarian vs schulze
- borda vs schulze

### 5.2 Reference-family hypotheses

Reference comparisons are reported separately:

- canonical vs plurality
- canonical vs random

`approval` is not part of confirmatory or reference-family hypothesis claims.

## 6. Statistical test protocol

- unit of inference: matched-seed paired differences
- primary test: two-sided paired sign-flip permutation test
- effect reporting:
  - paired mean difference
  - paired median difference
  - 95% bootstrap confidence interval

Multiplicity policy:

- canonical confirmatory family: Holm correction across all canonical
  endpoint-contrast tests (`4 endpoints x 3 contrasts = 12 tests`)
- reference family: BH-q reporting within the reference-family test pool
- canonical and reference p-value pools remain separate

## 7. Figure and table contract

Figures:

- F1: primary metric trajectories (canonical) with reference overlays
- F2: paired endpoint comparison plots for canonical contrasts
- F3: canonical confirmatory effect forest
- F4: reference-family effect panel

Tables:

- T1: frozen run protocol and provenance
- T2: endpoint formulas and hypothesis mapping
- T3: endpoint summaries per rule, including descriptive companion summaries
- T4: canonical confirmatory test results
- T5: reference-family results
- T6: robustness summaries

## 8. Change control

- no endpoint formula changes after the final-run readout
- no family-role reclassification after freeze without an explicit v2 protocol
- revisions require a new version file such as `thesis_analysis_protocol_v2.md`
