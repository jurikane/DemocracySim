# Environment Dynamics

This document is the single reference for **world-evolution semantics**:
how grid colors are initialized and mutated, how global color distributions are
computed, and which invariants are locked by tests for thesis runs.

## Where It Happens (Runtime Path)

Environment dynamics happen in two phases:

1. Initialization:
   - sample initial colors from `_preset_color_dst`
   - optional patching via `adjust_color_pattern(color_patches_steps, patch_power)`
2. Per-step execution:
   - `CustomScheduler.step()` applies mutation at the start of step `t+1`
   - areas execute elections on the election-time state
   - `update_global_color_distribution()` updates `global_color_dst`

Code references:

- `src/models/participation_model.py::ParticipationModel.create_color_distribution`
- `src/models/participation_model.py::ParticipationModel.adjust_color_pattern`
- `src/models/participation_model.py::CustomScheduler.step`
- `src/agents/area.py::Area.mutate_cells`
- `src/models/participation_model.py::ParticipationModel.update_global_color_distribution`

## Environment Semantics (Authoritative)

### Mutation timing

- Mutation from election step `t` is applied at the **start of step `t+1`**.
- Elections and logged step-state are evaluated on election-time (pre-mutation-of-`t`) state.

### Global color distribution

- `global_color_dst` is the normalized color distribution of the realized grid state.
- `update_global_color_distribution()` must be exact:
  - if areas are disjoint, it may use cached per-area counts (+ cached uncovered-cell counts)
  - otherwise it falls back to full grid counting

## Knobs and Contracts

### `mu`

- Mutation rate in `[0,1]` (fraction of an area’s cells recolored per mutation event).

### `election_impact_on_mutation`

- Finite `>= 0`.
- Shapes `color_probs` used to sample colors from elected orderings.
- `0` implies uniform probability over elected-ordering positions.

### `num_colors`

- Integer `>= 2`, with fail-loud cap via factorial option-space bound.
- Defines both grid color alphabet and election option dimensionality.

### `heterogeneity`

- Finite `>= 0`.
- Controls spread of the preset initialization distribution.
- `0` implies uniform preset distribution.

### `color_patches_steps`

- Integer `>= 0`.
- Number of full-grid patching passes at initialization.
- `0` is a no-op.

### `patch_power`

- Finite `>= 0`.
- Controls branch behavior in patching (`preset distribution` vs `neighbor consensus`).

## Why This Matters for Thesis Validity

- Environment dynamics define the outcome process that drives rewards and learning.
- Any silent drift in mutation timing or global-color computation can invalidate time-series interpretation.
- Strong contracts on these knobs are required for reproducible, defensible comparisons across voting rules.

## Logging and Integration Expectations

- Logged `color_*` series must match election-time grid semantics.
- `global_color_dst` used by learning/satisfaction paths must match realized grid counts.
- Topology-dependent fast paths must preserve exactness (no approximation drift).

## Test Coverage (What Is Locked By Pytests)

Core contracts:

- `tests/test_mu_mutation_contract.py`
- `tests/test_election_impact_on_mutation_contract.py`
- `tests/test_num_colors_contract.py`
- `tests/test_heterogeneity_contract.py`
- `tests/test_color_patches_steps_contract.py`
- `tests/test_patch_power_contract.py`
- `tests/test_global_color_distribution_semantics_contract.py`
- `tests/test_step_semantics_mutation_timing.py`
  - locks mutation timing semantics (`t` mutation applied at start of `t+1`)

Interaction test:

- `tests/test_environment_dynamics_interactions.py`
  - checks end-to-end environment pipeline (init distribution -> patching -> mutation timing -> global distribution consistency)

Related logging consistency guard:

- `tests/test_area_steps_pre_mutation_snapshot.py`
  - verifies area-level logged color distributions are pre-mutation election-time snapshots

## Recommended Thesis Run Policy

For baseline experiment grids:

- keep `num_colors` and mutation knobs fixed across rule comparisons
- use moderate `mu` and `election_impact_on_mutation` chosen in calibration, then freeze
- treat changes in `heterogeneity`/patching as explicit robustness analyses, not baseline variations
