# Technical Overview

**DemocracySim** is a multi-agent simulation framework for studying how voting rules shape participation, inequality, and outcome dynamics.

## First Steps

- `python -m scripts.run --config configs/default.yaml` launches the interactive Mesa server.
- `python -m scripts.run_headless --config configs/default.yaml` runs a headless batch.
- `python -m scripts.run_replay <run_dir>` opens a replay for a stored run directory.

## Scope

- Agents have limited information, preferences, assets, and adaptive behavior.
- Areas run elections under configurable voting rules.
- The environment mutates over time.
- Outputs are written in a structured format for reproducible analysis.

## Implemented Voting Rules

- `plurality_rule`
- `approval_voting`
- `utilitarian_rule`
- `borda_rule`
- `schulze_rule`
- `random_rule`

## Main Runtime and Analysis Entry Points

- `python -m scripts.run` (interactive Mesa server)
- `python -m scripts.run_headless` (batch/headless simulation)
- `python -m scripts.run_replay` (replay from stored run artifacts)
- `python -m scripts.run_doe` (design-of-experiments execution)
- `python -m scripts.score_doe` (DOE scoring and ranking)
- `python -m scripts.generate_summary` (run-level summary artifacts)

## Related Technical Pages

- `docs/technical/semantics_representation_rng.md`
- `docs/technical/voting_rules.md`
- `docs/technical/core_mechanics.md`
- `docs/technical/environment_dynamics.md`
- `docs/technical/structural_topology.md`
- `docs/technical/population_preferences.md`
- `docs/technical/participation_learning.md`
- `docs/technical/altruism_learning.md`
- `docs/technical/run_control_output.md`
