# Thesis Scope and Experimental Baseline

Research question:
ow do different voting rules influence the temporal evolution of participation rates and inequality in a simple multi-agent system with adaptive agents?

Related documents:

- `docs/research/execution_scope_freeze.md`
- `docs/research/thesis_model_concepts.md`

## 1. Scientific Scope

The thesis studies a fixed simulation environment where voting rule is the primary intentionally varied factor in confirmatory rule-comparison runs.

Participation and altruism are adaptive, but their update mechanisms are fixed within a given experiment set.

## 2. Baseline Model Settings

Current baseline family:

- `quality_target_mode = puzzle`
- `participation_signal_mode = group_relative_delta_rel_party`
- `altruism_mode = satisfaction`
- satisfaction-response parameters:
  - `altruism_satisfaction_theta`
  - `altruism_satisfaction_slope`
  - `altruism_response_gamma`

Implemented voting rules:

- `0 = majority`
- `1 = approval`
- `2 = utilitarian`
- `3 = borda`
- `4 = random` (reference arm)

## 3. Primary Outcomes

Primary outcome families are evaluated as time-series and run-level summaries:

- participation dynamics (`turnout`)
- resource inequality (`gini_assets` from `steps.gini_index`)
- experiential inequality (`gini_dissatisfaction` from `agents.dissatisfaction_value`)
- outcome-quality trajectory (`dist_to_reality` aggregated from `area_steps`)

Interpretation note:

- `assets` denotes simulation resource/capacity state, not literal income.

## 4. Secondary Descriptive Analyses

Secondary (non-confirmatory) views include:

- group-level turnout and participation-composition dynamics
- participation/inequality co-movement patterns

## 5. Out of Scope for Confirmatory Claims

- strategic voting equilibria
- empirical calibration against real election datasets
- normative policy prescriptions
- claims of globally optimal democratic design
