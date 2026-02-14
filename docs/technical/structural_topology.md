# Structural Topology

This document is the single reference for **area topology semantics**:
how areas are instantiated, what `no_overlap` means, and which invariants are
locked by tests for thesis runs.

## Where It Happens (Runtime Path)

Topology is fixed during model initialization:

1. `ParticipationModel.__init__` validates topology knobs.
2. `initialize_all_areas()` places all areas (regular anchors + optional additional anchors).
3. `Area.idx_field` builds each area’s cell membership.
4. `_analyze_area_coverage()` computes geometry facts:
   - covered cell count
   - disjointness (`_areas_are_disjoint`)
   - uncovered static color counts
   - finalized `no_overlap` semantics

Code references:

- `src/models/participation_model.py::ParticipationModel.__init__`
- `src/models/participation_model.py::ParticipationModel.initialize_all_areas`
- `src/models/participation_model.py::ParticipationModel._analyze_area_coverage`
- `src/agents/area.py::Area.idx_field`

## Topology Semantics (Authoritative)

### `no_overlap`

`model.no_overlap` means: **areas are disjoint** (no cell belongs to more than one area).

It does **not** imply full coverage of the grid.
Disjoint layouts with uncovered cells (gaps) still have `no_overlap=True`.

### Partition vs Disjoint

- `partition`: every grid cell belongs to exactly one area.
- `disjoint`: no double membership; uncovered cells may exist.

This distinction matters for interpretation and for optimization choices.

## Knobs and Contracts

### `num_areas`

- Must be integer `>= 1`.
- Must be `<= width * height` (unique anchor-slot bound).
- Valid input creates exactly `num_areas` areas.

### `av_area_height`, `av_area_width`

- Must be integer `>= 1`.
- Must be `<= grid` height/width respectively.

### `area_size_variance`

- Must be finite in `[0,1]`.
- Area dimensions are clamped to at least `1x1` in `Area._set_dimensions`.

## Why This Matters for Thesis Validity

- Topology affects local electorates, mutation locality, and logged area-level outcomes.
- Silent geometry errors can invalidate participation/inequality comparisons across runs.
- Fail-loud validation avoids ambiguous runtime failures and hidden invalid configurations.

## Logging and Integration Expectations

- `area_steps.parquet` must contain one row per real area per step.
- `steps.turnout` must equal mean of area-level turnout in `area_steps`.
- Topology mode (partition vs overlap) must not break schema consistency.

## Test Coverage (What Is Locked By Pytests)

Core contracts:

- `tests/test_num_areas_contract.py`
  - `num_areas` validation and exact count behavior

- `tests/test_area_geometry_knobs_contract.py`
  - `av_area_height/width` validation and bounds
  - `area_size_variance` validation and no-zero-sized-area guarantee
  - metamorphic effects of geometry knob changes

- `tests/test_overlap_semantics_contract.py`
  - `no_overlap` semantic oracle (disjointness, not partition)
  - metamorphic topology change assertions
  - integration check against `steps` color extraction path

Interaction test:

- `tests/test_structural_topology_interactions.py`
  - compares exact-partition topology vs overlapping topology
  - verifies structural invariants in both
  - verifies logging consistency (`steps` and `area_steps`) in both

Additional regression guard:

- `tests/test_overlap_runtime_regression.py`
  - ensures overlapping topologies can execute multiple steps and produce valid v2 logs

## Recommended Thesis Run Policy

For baseline experiment grids:

- prefer partition-like setups (`no_overlap=True` and full coverage) unless overlap is a deliberate treatment variable
- keep `area_size_variance=0` for baseline comparability
- vary topology only in dedicated robustness checks

## Thesis Scope Note

Overlapping areas are supported by the framework and remain regression-tested.
They are intentionally not used as part of the thesis baseline experiment design.
