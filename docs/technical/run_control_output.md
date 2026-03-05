# Run Control & Output

This page documents how to run simulations, DOE workflows, and summary generation.

## Main Commands

### Single run (UI)

```bash
python -m scripts.run --config configs/default.yaml
```

### Headless batch run

```bash
python -m scripts.run_headless --config configs/default.yaml
```

### Replay an existing run

```bash
python -m scripts.run_replay <run_dir>
```

### Generate run summary artifacts

```bash
python -m scripts.generate_summary --run-dir <run_dir>
```

Optional:

```bash
python -m scripts.generate_summary --run-dir <run_dir> --mode fast --closed
```

### Run DOE

```bash
python -m scripts.run_doe --config doe.yaml --doe-profile <profile> --points <n>
```

Common seed options:

```bash
python -m scripts.run_doe --seed-mode fixed --seeds 101,202,303
python -m scripts.run_doe --seed-mode stratified --seed-target 75 --seed-candidate-start 100 --seed-candidate-count 300
```

### Score DOE

```bash
python -m scripts.score_doe --doe-root data/simulation_output/doe_<timestamp>
```

Optional objective override:

```bash
python -m scripts.score_doe --doe-root data/simulation_output/doe_<timestamp> --objective-config configs/doe_selection_objective_v1.json
```

## DOE Support Tools

### Build HIL queue

```bash
python -m tools.doe.build_doe_hil_queue --doe-root data/simulation_output/doe_<timestamp>
```

### Review HIL queue

```bash
python -m tools.doe.doe_hil_review --print-commands
python -m tools.doe.doe_hil_review --populate-ai
python -m tools.doe.doe_hil_review --bucket top --limit 5 --run-fast
```

### Probe scoring dimensions

```bash
python -m tools.doe.probe_scoring_dimensions --doe-root data/simulation_output/doe_<timestamp>
```

### Recovery scan

```bash
python -m tools.doe.recovery_scan --doe-root data/simulation_output/doe_<timestamp>
```

### Seed selection helper

```bash
python -m tools.doe.select_balanced_seeds --config doe.yaml --doe-profile <profile> --target <n>
```

### DOE inference report

```bash
python -m tools.research.doe_inference --doe-root data/simulation_output/doe_<timestamp>
```

## Output Structure (Headless/DOE)

Typical run output root:

- `data/simulation_output/<run_or_doe_root>/`

Common files:

- `config_used.yaml`
- `meta.yaml`
- `static.json`
- `agents.parquet`
- `steps.parquet`
- `area_steps.parquet`
- `votes.parquet`
- `analysis/` (summary CSV/JSON/PDF artifacts)

## Determinism and Seeds

- Per-run seed: `run_seed = base_seed + run_id`
- Reproducibility requires same code, config, seed, and run length.

## Practical Workflow

1. Run simulations or DOE.
2. Score DOE outputs.
3. Inspect top/mid/bottom designs with HIL queue + summaries.
4. Freeze selected configuration and generate thesis runs.
