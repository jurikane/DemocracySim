# Thesis Contract: Voting Rules, Participation, and Inequality Dynamics

Research question:
How do different voting rules influence the temporal evolution of participation rates and inequality in a simple multi-agent system with adaptive agents?

Execution-level freeze and gate control:

- `docs/research/execution_scope_freeze.md`

Concept model documentation:

- `docs/research/thesis_model_concepts.md`

## Current Contract (Implemented Truth)

### Scientific scope

This thesis studies a fixed simulation environment where only the voting-rule arm is intentionally varied for rule-comparison runs. The model includes adaptive participation behavior and adaptive altruism under a fixed update regime.

### Baseline mechanics currently implemented

- Quality target mode baseline: `quality_target_mode=puzzle`
- Participation signal regime baseline: `participation_signal_mode=group_relative_delta_rel_party`
- Altruism baseline family: `altruism_mode=satisfaction`
  - with satisfaction-response parameters (`altruism_satisfaction_theta`, `altruism_satisfaction_slope`) and response gain (`altruism_response_gamma`)
- Implemented voting rules (rule index contract):
  - `0=majority`
  - `1=approval`
  - `2=utilitarian`
  - `3=borda`
  - `4=random` (reference arm)

### Outcome focus (current)

Primary thesis outcome families are tracked as time series and run-level summaries:

- Participation dynamics (`turnout`)
- Resource inequality dynamics (`gini_assets` from `steps.gini_index`)
- Experiential inequality dynamics (`gini_dissatisfaction` from `agents.dissatisfaction_value`)
- Outcome quality trajectory (`dist_to_reality` aggregated from `area_steps`)

Interpretation constraint:

- `assets` is simulation resource/capacity state, not literal income.

Secondary descriptive lenses (non-confirmatory):

- group-level turnout and participation-composition diagnostics (majority/minority dynamics by personality group)
- participation and inequality co-movement diagnostics across time

### Exclusions (current)

Out of scope for confirmatory claims:

- strategic voting mechanisms
- empirical validation against real election data
- normative policy prescription
- strong claims about optimal democratic design

## Freeze-Target Contract (Decided, May Include Pending Items)

### Voting-rule arm design

- Keep canonical implemented arm family as the primary confirmatory family.
- Keep `random` voting rule as a reference arm:
  - semantics: uniform full random ranking per election
  - deterministic replay under fixed seed
  - role: reference family (non-confirmatory unless reclassified later)

### Independent-variable discipline

For final thesis experiments, voting rule remains the only intentionally varied independent variable in the confirmatory comparison matrix.

### Contract layering rule

Core docs are allowed to include freeze-target items before implementation, but every non-implemented item must be explicitly marked pending with dependency tracking.

## Implementation Status Notes

- Random reference arm is implemented in runtime, metadata, and analysis paths.
- Core doc/summary drift checks are implemented for current summary-key coverage.
- Final D0 thesis-lead sign-off is tracked in internal freeze records before Gate C closure.
