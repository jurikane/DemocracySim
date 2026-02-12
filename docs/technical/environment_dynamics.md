# Environment Dynamics

This page documents the **environment / world evolution knobs** and the
semantics that are locked by Pytests. These knobs affect the evolution of the
grid (color cells), which in turn changes election outcomes and reward signals.

## Core Knobs

### `mu` (Mutation Rate)

- Meaning: fraction of an area's cells that are recolored when mutation is applied.
- Range: `[0, 1]`.
- Timing: mutation from step `t` is applied at the **start of step `t+1`** (see step semantics).

Locked by:

- `tests/test_mu_mutation_contract.py`

### `election_impact_on_mutation` (Bias Shape for Mutation Sampling)

- Meaning: shapes the probability vector `color_probs` used when sampling colors from
  the elected ordering during mutation.
- Range: finite `>= 0`.
- Special case: `0` means uniform sampling across the elected ordering.

Locked by:

- `tests/test_election_impact_on_mutation_contract.py`

### `num_colors` (Option Space Size)

- Meaning: number of colors in the grid, and the number of alternatives in elections.
- Constraint: the model uses all **permutations** of colors as options; this grows as `num_colors!`.
- Policy: fail loudly for configurations that would imply a huge option space.

Locked by:

- `tests/test_num_colors_contract.py`

### `heterogeneity` (Preset Distribution Sharpness)

- Meaning: controls the variability of the initial preset distribution used to sample
  initial cell colors.
- Range: finite `>= 0`.
- Special case: `0` implies a uniform preset distribution.

Locked by:

- `tests/test_heterogeneity_contract.py`

### `color_patches_steps` (Initialization-Only Patching Passes)

- Meaning: number of full-grid “patching” passes applied after initial cell creation.
- Range: integer `>= 0`.
- Special case: `0` disables patching entirely (no-op).

Locked by:

- `tests/test_color_patches_steps_contract.py`

### `patch_power` (Initialization Patching Strength)

- Meaning: controls the strength of patching (via the Gaussian term in `color_patches()`),
  i.e. how often patching draws from the preset distribution vs. following neighbor consensus.
- Range: finite `>= 0`.
- Special case: `0` forces the preset-distribution branch for any cell with non-zero bias distance.

Locked by:

- `tests/test_patch_power_contract.py`

## Global Color Distribution Semantics

The model maintains:

- `global_color_dst`: the normalized global distribution of colors on the grid (election-time state)
- `update_global_color_distribution()`: updates `global_color_dst`

**Definition:** `global_color_dst` must match the realized grid state (count colors across all cells and normalize).

Performance optimization:

- If areas are **disjoint**, global counts can be computed exactly by summing cached per-area color counts,
  plus a cached contribution from uncovered (static) cells.
- If areas overlap, the model falls back to scanning the grid (can be improved in future).

Locked by:

- `tests/test_global_color_distribution_semantics_contract.py`

## Interaction Test (Pipeline Sanity)

Because these knobs interact (init distribution → patching → mutation → global distribution),
we also lock a small end-to-end environment pipeline test that avoids coupling to the
full learning/voting system by stubbing `Area.step()`.

Locked by:

- `tests/test_environment_dynamics_interactions.py`
