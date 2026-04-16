# Structural Topology

This page describes how areas are placed on the grid and what the main topology
settings mean. These choices affect how elections are localized and how much
territorial structure the model has.

## Implementation Reference

Area placement and coverage analysis are documented in
[ParticipationModel](api/Model.md). Area-local indexing and territory behavior
are documented in [Area](api/Area.md).

## Key Semantics

### `no_overlap`

`no_overlap=True` means areas are disjoint: no cell belongs to more than one area.
It does not imply full grid coverage.

### Partition vs Disjoint

- `partition`: every cell belongs to exactly one area.
- `disjoint`: no double membership; uncovered cells are allowed.

## Main Knobs

- `num_areas` (`>= 1`, bounded by grid size)
- `av_area_height` (`>= 1`, bounded by grid height)
- `av_area_width` (`>= 1`, bounded by grid width)
- `area_size_variance` (`[0,1]`)
