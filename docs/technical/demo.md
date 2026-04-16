# Demo

This is the quickest local path into DemocracySim. The demo config is small
enough to run in seconds, but still shows the main feedback loop between
elections, participation, inequality, and changing grid conditions.

![DemocracySim demo teaser](../images/demo/demo_teaser.gif)

## Demo Config

[configs/demo.yaml](https://github.com/jurikane/DemocracySim/blob/main/configs/demo.yaml)
uses a compact setup:

- 50 agents
- 4 colors
- 4 preference groups
- 2 areas
- one headless run with replay output

## Generate And Replay

Create a local environment once:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate a demo run:

```bash
python -m scripts.run_headless --config configs/demo.yaml --out-root tmp/demo_gif_run
```

Replay the stored run:

```bash
python -m scripts.run_replay tmp/demo_gif_run/run_0
```

Or start an un-seeded live demo:

```bash
python -m scripts.run --config configs/demo.yaml
```

## What To Look At

The grid is the current election-time world state. It is not static: collective
decisions influence later mutation, so the grid stores part of the system's
history.

The main plots to watch are:

- turnout
- asset inequality and dissatisfaction inequality
- outcome quality and group distance to the elected outcome

For the quality measure used here, see
[Puzzle Quality Gate](../research/puzzle_quality_gate_concept.md).

![DemocracySim demo preview](../images/demo/demo_view_step_215.webp)

## Replay Your Own Run

Replay is deterministic because it reads recorded run artifacts rather than
resimulating the model.
Replay any stored run with:

```bash
python -m scripts.run_replay <run_dir>
```

Or search for runs interactively in `data/`:

```bash
python -m scripts.run_replay
```

Generate summary artifacts for a stored run:

```bash
python -m scripts.generate_summary --run-dir <run_dir>
```
