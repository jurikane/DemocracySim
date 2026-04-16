# Tools

This folder contains non-core CLI utilities used for analysis, diagnostics, and research support.

## Boundary

- Use `scripts/` for stable pipeline entrypoints (`run_doe`, `score_doe`, `generate_summary`, etc.).
- Use `tools/` for helper workflows that support exploration, validation, and debugging.

## Subfolders

- `tools/doe/`: DOE helper utilities (review bundles, probing, seed tooling, sanity matrix, inference).

## Stability

- Tools are expected to be useful and tested where practical.
- CLI interfaces may evolve faster than `scripts/` interfaces.
