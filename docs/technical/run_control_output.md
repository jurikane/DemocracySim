# Run Control & Output

This page collects the main runtime entry points, the output structure of a
stored run, and the basic reproducibility conventions used by the project.

## Main Runtime Paths

Start the interactive Mesa server:

```bash
python -m scripts.run --config configs/default.yaml
```

Run a headless batch:

```bash
python -m scripts.run_headless --config configs/default.yaml
```

Replay a stored run:

```bash
python -m scripts.run_replay <run_dir>
```

Generate run-level summary artifacts:

```bash
python -m scripts.generate_summary --run-dir <run_dir>
```

A faster summary variant is also available:

```bash
python -m scripts.generate_summary --run-dir <run_dir> --mode fast --closed
```

## Voting Rule Selection In Config

The active aggregation rule is selected by `model.rule_idx`:

- `0 = plurality`
- `1 = approval`
- `2 = utilitarian`
- `3 = borda`
- `4 = schulze`
- `5 = random`

For what these rules mean conceptually and operationally, see
[voting_rules.md](voting_rules.md).

## DOE Entry Points

Run a DOE batch:

```bash
python -m scripts.run_doe --config doe.yaml --doe-profile <profile> --points <n>
```

Common seed options:

```bash
python -m scripts.run_doe --seed-mode fixed --seeds 101,202,303
python -m scripts.run_doe --seed-mode stratified --seed-target 75 --seed-candidate-start 100 --seed-candidate-count 300
```

Score a DOE result set:

```bash
python -m scripts.score_doe --doe-root data/simulation_output/doe_<timestamp>
```

Optional objective override:

```bash
python -m scripts.score_doe --doe-root data/simulation_output/doe_<timestamp> --objective-config configs/doe_selection_objective_v1.json
```

Additional DOE tools:

```bash
python -m tools.doe.select_balanced_seeds --config doe.yaml --doe-profile <profile> --target <n>
python -m tools.doe.doe_inference --doe-root data/simulation_output/doe_<timestamp>
```

## Output Structure

Headless and DOE runs write into a run root under `data/simulation_output/`
unless an explicit output path is supplied.

A typical stored run contains:

- `config_used.yaml`
- `meta.yaml`
- `static.json`
- `agents.parquet`
- `steps.parquet`
- `area_steps.parquet`
- `votes.parquet`
- `analysis/` for derived CSV, JSON, and PDF summary artifacts

The schema records consistent step timing, rule labels, and run metadata so
that replay, summaries, and downstream analysis can read the same run
artifact set.

## Determinism And Seeds

Per-run seeding follows:

- `run_seed = base_seed + run_id`

Reproducibility therefore requires the same code, config, seed, and run length.
Replay is deterministic because it reads the stored artifacts rather than
resimulating the election path.

## Practical Workflow

The normal workflow is:

1. run a live or headless simulation
2. replay or inspect the stored run
3. generate summary artifacts if needed
4. for DOE work, score the result set and continue with selected designs
