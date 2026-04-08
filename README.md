[![Pages](https://github.com/jurikane/DemocracySim/actions/workflows/ci.yml/badge.svg)](https://jurikane.github.io/DemocracySim/)
[![pytest main](https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/jurikane/DemocracySim/branch/main/graph/badge.svg)](https://codecov.io/gh/jurikane/DemocracySim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# DemocracySim: Multi-Agent Simulation of Voting Rules, Participation, and Inequality

DemocracySim is a Mesa-based research platform for studying how voting rules shape
participation, inequality, and collective outcomes in a dynamic multi-agent
environment.

The project started as a master's thesis and continues as the public branch for
further experiments and development. The thesis freeze is kept separately on
branch `thesis`.

The associated thesis asks:

> How do different voting rules influence the temporal evolution of
> participation rates and inequality in a simple multi-agent system with
> adaptive agents?

This project was supported by [OpenPetition](https://osd.foundation),
and developed in the [Swarm Intelligence and Complex Systems](https://siks.informatik.uni-leipzig.de)
group at [Leipzig University](https://www.uni-leipzig.de/en).

## Start Here

Documentation:

- [Project docs on GitHub Pages](https://jurikane.github.io/DemocracySim/)
- [Technical overview](docs/technical/technical_overview.md)
- [Voting rules](docs/technical/voting_rules.md)

---

Create a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the interactive Mesa server:

```bash
python -m scripts.run --config configs/default.yaml
```

Run a headless batch from the default config:

```bash
python -m scripts.run_headless --config configs/default.yaml
```

Replay an existing stored run:

```bash
python -m scripts.run_replay <run_dir>
```

## What This Repository Contains

- interactive simulation via Mesa
- headless and batch execution pipelines
- replay tooling for stored run artifacts
- configurable voting rules and adaptive participation behavior
- structured outputs for summaries, diagnostics, and further analysis

## Model Snapshot

Agents live in a grid-based environment that evolves over time. Areas hold
repeated elections, and agents decide whether to participate or abstain before
their preferences are aggregated under a voting rule.

Election outcomes affect both rewards and the subsequent state of the
environment, creating feedback between collective decisions and later
conditions. The framework is designed to make these dynamics inspectable rather
than hiding them behind a single end metric.

Core ingredients:

- heterogeneous agent preferences
- adaptive participation behavior
- multiple voting rules
- path-dependent environmental change
- structured logging for replay and downstream analysis

## Thesis Context

The thesis focuses on a controlled subset of this broader framework. It compares
how different voting rules affect turnout and inequality under fixed model
assumptions.

The public `main` branch keeps that context visible, but it is not the archival
freeze or the university hand-in branch. It should remain useful for further
experiments, cleanup, and public exploration of the project.

## Out of Scope for the Thesis

- strategic voting equilibria
- empirical calibration to real election data
- policy recommendations
- claims about globally optimal democratic design

These remain possible directions for future work rather than claims made by the
thesis itself.
