# Tools

This folder contains non-core CLI utilities used for analysis, diagnostics, and research support.

## Boundary

- Use `scripts/` for stable, thesis-facing pipeline entrypoints (`run_doe`, `score_doe`, `generate_summary`, etc.).
- Use `tools/` for helper workflows that support exploration, validation, and debugging.

## Subfolders

- `tools/doe/`: DOE helper utilities (HIL queues/review, probing, recovery scans, seed tooling, sanity matrix).

## Stability

- Tools are expected to be useful and tested where practical.
- CLI interfaces may evolve faster than `scripts/` interfaces.
