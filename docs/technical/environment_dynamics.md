# Environment Dynamics

The grid is the realized world state of the model, so its dynamics matter both
for what agents currently observe and for what later elections inherit from
earlier ones.

## Runtime Flow

Initialization starts from a preset color distribution and can then be locally
patched with `color_patches_steps` and `patch_power` so the starting grid is
less spatially uniform.

After that, each step follows the same timing. Mutation from election `t` is
applied at the start of step `t+1`, elections run on the current election-time
state, and `global_color_dst` is updated from the realized grid.

## Implementation Reference

Initialization, scheduler timing, and global grid updates are documented in
[ParticipationModel](api/Model.md). Area-level mutation logic is documented in
[Area](api/Area.md).

## Main Knobs

- `mu` in `[0,1]`: fraction of area cells recolored per mutation event
- `election_impact_on_mutation` `>= 0`: how strongly elected orderings shape mutation probabilities
- `num_colors` `>= 2`: color alphabet and election option count
- `heterogeneity` `>= 0`: spread of the preset initialization distribution
- `color_patches_steps` `>= 0`: number of initialization patching passes
- `patch_power` `>= 0`: patching preference for local consensus versus preset distribution
