# Thesis Scope and Experimental Baseline

Research question:
How do different voting rules influence the temporal evolution of participation rates and inequality in a simple multi-agent system with adaptive agents?

Related documents:

- `docs/research/thesis_measurement_spec.md`
- `docs/research/metric_glossary.md`
- `docs/research/thesis_model_concepts.md`

## 1. Scientific Scope

The thesis studies a fixed simulation environment where voting rule is the primary intentionally varied factor in rule-comparison runs.

Participation and altruism are adaptive, but their update mechanisms are fixed within a given experiment set.

## 2. Baseline Model Settings

Baseline family:

- `quality_target_mode = puzzle`
- `participation_signal_mode = group_relative_delta_rel_party`
- `altruism_mode = satisfaction`
- satisfaction-response parameters:
  - `altruism_satisfaction_theta`
  - `altruism_satisfaction_slope`
  - `altruism_response_gamma`

Implemented voting-rule mapping (`rule_idx`):

- `0 = majority`
- `1 = approval`
- `2 = utilitarian`
- `3 = borda`
- `4 = schulze`
- `5 = random`

## 3. Rule-Family Framing for Thesis Inference

- confirmatory canonical family: `utilitarian (2)`, `borda (3)`, `schulze (4)`
- reference family: `majority (0)`, `random (5)`
- context-only calibration arm: `approval (1)`

Family labels define reporting roles, not implementation differences in runtime mechanics.

## 4. Primary Outcomes

Primary outcome families are evaluated as time-series and run-level summaries:

- participation dynamics (`turnout`)
- resource inequality (`gini_assets` from `steps.gini_index`)
- experiential inequality (`gini_dissatisfaction` from `agents.dissatisfaction_value`)
- outcome-quality trajectory (`quality_distance` aggregated from `area_steps` with mode-aware source)

Interpretation note:

- `assets` denotes simulation resource/capacity state, not literal income.

## 5. Secondary Descriptive Analyses

Secondary (non-confirmatory) views include:

- group-level turnout and participation-composition dynamics
- participation/inequality co-movement patterns
- benchmark-reference distance diagnostics from logged artifacts

## 6. Out of Scope for Confirmatory Claims

- strategic voting equilibria
- empirical calibration against real election datasets
- normative policy prescriptions
- claims of globally optimal democratic design
