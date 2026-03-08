# Environment Dynamics

This page describes how grid states are initialized and mutated over time.

## Runtime Flow

1. Initialization
- Sample initial colors from preset distribution.
- Optional patching with `color_patches_steps` and `patch_power`.

2. Per-step execution
- Mutation from election `t` is applied at the start of step `t+1`.
- Elections run on the current election-time state.
- `global_color_dst` is updated from the realized grid.

## Runtime Locations

- `src/models/participation_model.py::ParticipationModel.create_color_distribution`
- `src/models/participation_model.py::ParticipationModel.adjust_color_pattern`
- `src/models/participation_model.py::CustomScheduler.step`
- `src/agents/area.py::Area.mutate_cells`
- `src/models/participation_model.py::ParticipationModel.update_global_color_distribution`

## Main Knobs

- `mu` in `[0,1]`: fraction of area cells recolored per mutation event.
- `election_impact_on_mutation` `>= 0`: how strongly elected orderings shape mutation probabilities.
- `num_colors` `>= 2`: color alphabet and election option count.
- `heterogeneity` `>= 0`: spread of the preset initialization distribution.
- `color_patches_steps` `>= 0`: number of initialization patching passes.
- `patch_power` `>= 0`: patching preference for local consensus versus preset distribution.
