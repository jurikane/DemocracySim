[![Pages](https://github.com/jurikane/DemocracySim/actions/workflows/ci.yml/badge.svg)](https://jurikane.github.io/DemocracySim/)
[![pytest main](https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/jurikane/DemocracySim/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/jurikane/DemocracySim/branch/main/graph/badge.svg)](https://codecov.io/gh/jurikane/DemocracySim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# DemocracySim: Multi-Agent Simulation of Voting Rules, Participation, and Inequality

Codebase for the master thesis

****"How do different voting rules influence the temporal evolution of participation rates and inequality in a simple multi-agent system with adaptive agents?"****

conducted at the group [Swarm Intelligence and Complex Systems](https://siks.informatik.uni-leipzig.de)
at the [Faculty of Mathematics and Computer Science](https://www.mathcs.uni-leipzig.de/en)
of [Leipzig University](https://www.uni-leipzig.de/en).

This project is kindly supported by [OpenPetition](https://osd.foundation).

## Documentation

Additional [documentation](https://jurikane.github.io/DemocracySim/) on GitHub-pages.

### Reproducibility

Public reproducibility scope, artifacts, and commands are documented in:

- `docs/technical/reproducibility.md`
- `docs/technical/hand_in_guide.md`
- `docs/technical/final_analysis_package.md`

Quick checks:

```bash
PYTHONPATH=. python scripts/repro/verify_thesis_repro_bundle.py
PYTHONPATH=. python scripts/repro/audit_doe_design_lock.py
PYTHONPATH=. python scripts/run_final_manifest.py --dry-run
PYTHONPATH=. python scripts/build_thesis_analysis_package.py --dry-run
```

---

## Thesis Scope

The master thesis associated with this repository focuses on a *controlled subset* of the simulation framework.

Specifically, the thesis investigates how different **voting rules** influence the **temporal evolution of participation rates and inequality** in a simple multi-agent simulation with adaptive agents. The analysis is based on time-series data generated under fixed environmental and behavioral assumptions, comparing outcomes across a small number of canonical voting rules.

While the codebase supports additional agent behaviors, metrics, and normative evaluation criteria, these features are **explicitly out of scope for the thesis contribution** and are considered extensions for future research.

---

## Overview

**DemocracySim** is a multi-agent simulation framework designed to study democratic participation and collective decision-making in a controlled, evolving environment.

Agents are situated within a grid-based world and repeatedly participate in elections that aggregate individual preferences into collective decisions.
These decisions affect both the distribution of rewards among agents and the subsequent evolution of the environment, creating feedback between individual behavior and collective outcomes.

The environment is implemented as a toroidal grid of colored fields, where neighboring groups of cells form territories.
Each territory holds regular elections in which agents vote on the observed color distribution.
Election outcomes influence agent rewards and drive controlled mutation processes that update the environment over time.

Agents have limited resources and heterogeneous preferences over current color distributions as well as possible election outcomes ("personalities").
At each election, agents decide whether to participate or abstain, creating a participation dilemma.
When voting, agents face a trade-off between aligning with their personal preferences and contributing to collective accuracy, as collective decisions affect future rewards.

---

## Agents

Agents are heterogeneous and bounded in their decision-making capabilities. Each agent:

- Possesses preferences over possible outcomes (personality types)
- Has limited resources that evolve over time
- Decides whether to participate in elections
- Adapts its behavior based on experienced outcomes via a fixed learning mechanism

Personality types are distributed across the population to induce majority–minority situations, but the thesis does not focus on group-specific optimization or strategic behavior.

---

## Elections and Voting Rules

Elections aggregate individual agent inputs into collective decisions using predefined **voting rules**.
Voting rules are the primary experimental manipulation in the thesis.

The thesis compares a small set of canonical voting rules while keeping all other model components constant.
Elections determine collective outcomes that influence reward allocation and environmental updates.

---

## Metrics and Data Collection

The simulation infrastructure supports the collection of a wide range of behavioral and system-level metrics.

The **core thesis analysis** focuses on:

- **Participation rate**: the proportion of agents participating in elections over time
- **Asset inequality**: measured using the Gini index over agent resources
- **Dissatisfaction inequality**: measured using the Gini index over agent dissatisfaction
- **Quality distance**: the distance between collective outcomes and the active puzzle target

Additional summaries such as average dissatisfaction, collective asset levels, or auxiliary run diagnostics may be used descriptively, but they are not part of the main confirmatory endpoint set.

---

## Out of Scope (Thesis)

The following aspects are explicitly **out of scope for the master thesis**, even if partially supported by the codebase:

- Strategic voting or game-theoretic equilibrium analysis
- Complex or multi-stage learning mechanisms
- Empirical validation or real-world policy recommendations
- Normative evaluation frameworks (e.g. utilitarian, egalitarian, Rawlsian optimization)
- Claims about collective intelligence or optimal democratic design

These aspects are considered directions for future research beyond the thesis.

---

## Project Vision (Beyond the Thesis)

Beyond the scope of the master thesis, **DemocracySim** is intended as a flexible research platform for exploring more complex questions related to collective decision-making, participation, fairness, and democratic system design.
Potential future extensions include richer agent models, alternative decision-making mechanisms, and applications to real-world collaborative or political settings.
