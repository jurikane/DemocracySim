# Technical Overview

DemocracySim is a Mesa-based simulation framework for studying how voting rules
shape participation, inequality, and collective outcomes over time. It is not a
single-shot voting model: elections feed back into rewards, learning, and later
environmental change.

This page is the technical entry point for the project. It gives a compact
picture of what the model does and where to continue reading.

## Core Simulation Loop

At each step, the simulation runs a repeated election process inside each area:

1. The current grid state defines the election-time world the agents face.
2. Agents update their local knowledge from either the realized area state or,
   in puzzle mode, from the area's current puzzle target.
3. Each eligible agent decides whether to participate.
4. Participants cast ballots, and the configured voting rule aggregates them
   into an elected color ordering.
5. The outcome is evaluated against a quality target, which determines whether
   the election produces rewards or penalties.
6. Asset changes and group-relative signals update later participation
   behavior.
7. The elected outcome influences later mutation of the grid, so future
   elections take place in a changed environment.

This closed loop is why voting rules matter here. Rule choice affects not only
one election result, but also later turnout, inequality, and the conditions
under which future elections happen.

## Main Entities And State

### Vote agents

`VoteAgent` instances carry:

- an ordinal preference-group ranking over colors
- an individual preference distribution consistent with that ranking
- assets
- local knowledge about the environment
- adaptive participation state
- optional altruism-related state

The distinction between preference groups and individual preference
distributions matters. A preference group fixes the ordering of colors, while
`personal_opt_dist` determines how strongly an agent favors the colors within
that ordering.

### Areas

`Area` instances are the local election arenas. They hold a set of agents and a
subset of the grid, run elections, compute local outcome quality, apply reward
and fee effects, and track area-level statistics.

### Grid and color cells

The grid is the realized world state. Its color distribution is path-dependent:
past election outcomes shape later mutation. In that sense, the grid stores
part of the model's political history.

### Model-level state

At model level, DemocracySim keeps the option space, the available voting
rules, the global color distribution, area structure, logging hooks, and the
configuration that determines how elections, learning, and mutation interact.

## Two Important Model Distinctions

### Preference structure

The model separates:

- shared preference groups
- per-agent preference intensity

This allows agents to agree on the order of colors while still differing in how
strongly they weight that order.

### Realized world state vs quality target

The model also separates:

- the realized grid state
- the target used to judge decision quality

In `quality_target_mode="reality"`, outcomes are judged against the current
realized state. In `quality_target_mode="puzzle"`, each area carries an
additional puzzle target, and outcome quality is judged against that target
instead. This distinction matters because an outcome can be group-favoring or
popular without necessarily being quality-improving.

## What Is Configurable

The main configuration surface includes:

- population size and preference-group structure
- grid size and area topology
- voting rule and ordering-distance function
- participation costs and reward strength
- quality-target mode and puzzle parameters
- participation learning parameters
- altruism and satisfaction parameters
- simulation length, output settings, and visualization settings

See:

- [population_preferences.md](population_preferences.md)
- [structural_topology.md](structural_topology.md)
- [voting_rules.md](voting_rules.md)
- [core_mechanics.md](core_mechanics.md)
- [participation_learning.md](participation_learning.md)
- [environment_dynamics.md](environment_dynamics.md)

## Running And Inspecting The Project

The three main entry points are:

- `python -m scripts.run` for the live Mesa server
- `python -m scripts.run_headless` for recorded batch runs
- `python -m scripts.run_replay <run_dir>` for replaying a stored run

Additional analysis entry points include:

- `python -m scripts.generate_summary`
- `python -m scripts.run_doe`
- `python -m scripts.score_doe`

For command usage and output layout, see
[run_control_output.md](run_control_output.md). For the compact example path,
see [demo.md](demo.md).

## Where To Read Next

If you want to understand the model mechanics first:

- [core_mechanics.md](core_mechanics.md)
- [participation_learning.md](participation_learning.md)
- [environment_dynamics.md](environment_dynamics.md)

If you want to understand representation and setup:

- [population_preferences.md](population_preferences.md)
- [structural_topology.md](structural_topology.md)
- [semantics_representation_rng.md](semantics_representation_rng.md)

If you want to understand aggregation and outputs:

- [voting_rules.md](voting_rules.md)
- [run_control_output.md](run_control_output.md)
