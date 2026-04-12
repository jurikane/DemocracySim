# Demo

This page is the fastest way to experience DemocracySim from the public branch.

## What The Bundled Demo Shows

The bundled demo is a tiny replayable Approval run with:

- 4 colors
- 4 areas
- 60 agents
- puzzle-based quality tracking
- full schema-v2 replay data

It is intentionally small so it works from a fresh clone without thesis data,
DOE bundles, or long preprocessing steps.

## Launch The Bundled Replay

Create a local environment and install the project once:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional faster setup with `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip sync requirements.txt
```

Start the public demo replay:

```bash
python -m scripts.run_replay --demo
```

For smoke checks or presentation fallback setups without an auto-opened tab:

```bash
python -m scripts.run_replay --demo --no-browser
```

## What To Look At In The Replay

- The grid shows the current election-time world state.
- Area panels expose turnout, quality distance, and the elected ordering.
- The replay is deterministic because it reuses recorded schema-v2 outputs.
- The bundled run is small enough to inspect quickly, but still shows the core
  feedback loop between voting, participation, and evolving local conditions.

## Replay Your Own Run

After generating a run locally, replay it with:

```bash
python -m scripts.run_replay <run_dir>
```

If you want summary PDFs and CSV sidecars for that run as well:

```bash
python -m scripts.generate_summary --run-dir <run_dir>
```

## Demo Asset

The bundled asset lives in:

`examples/demo_runs/approval_sparse_v1/run_0`

It is a curated public demo package, not a thesis freeze artifact and not a
benchmark reference bundle.
