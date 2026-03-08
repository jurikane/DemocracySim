# Structural Topology

This page describes how areas are placed on the grid.

## Runtime Locations

- `src/models/participation_model.py::ParticipationModel.__init__`
- `src/models/participation_model.py::ParticipationModel.initialize_all_areas`
- `src/models/participation_model.py::ParticipationModel._analyze_area_coverage`
- `src/agents/area.py::Area.idx_field`

## Key Semantics

### `no_overlap`

`no_overlap=True` means areas are disjoint (no cell belongs to more than one area).
It does not imply full grid coverage.

### Partition vs Disjoint

- `partition`: every cell belongs to exactly one area.
- `disjoint`: no double membership; uncovered cells are allowed.

## Main Knobs

- `num_areas` (`>= 1`, bounded by grid size)
- `av_area_height` (`>= 1`, bounded by grid height)
- `av_area_width` (`>= 1`, bounded by grid width)
- `area_size_variance` (`[0,1]`)
