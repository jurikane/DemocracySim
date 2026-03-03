# Technical overview

**DemocracySim** is a multi-agent simulation framework designed to examine democratic participation.
This project models agents (with personal interests forming majority-minority groups), environments
(evolving under the influence of the collective behavior of the agents),
and elections to analyze how voting rules influence participation,
welfare, system dynamics and overall collective outcomes.

Key features:

- Multi-agent system simulation using **Mesa framework**.
- **Grid-based environment** with wrap-around support (toroidal topology).
- Explore societal outcomes under different voting rules.

## Voting Rules (Primary Independent Variable)

The voting rule is the primary independent variable for thesis comparisons.
Implemented rule set:

- `majority_rule`
- `approval_voting`
- `utilitarian_rule`
- `borda_rule`
- `random_rule` (reference arm)

## Features

- **Agents**:
  - Independently acting entities modeled with preferences, budgets, and decision-making strategies.
  - Can participate in elections, have personal preferences and limited information about surroundings.
  - Core election economics are fee + common reward + personal reward with explicit break-even semantics.

- **Environment**:
  - Structured as a grid divided into "territories" or "areas."
  - A single unit of the grid is a "cell" or "field."
  - Each cell has a specific "color" representing a state. Elections influence these states, and areas mutate over time.
  - The initial grid can be made less i.i.d.-random via an initialization-only “patching” stage:
    `color_patches_steps` controls how many full-grid smoothing passes are applied (0 disables patching),
    and `patch_power` controls how strongly patching prefers local neighbor consensus (larger values)
    versus drawing colors from the preset distribution (smaller values).
  - `global_color_dst` is the (normalized) global color distribution computed from the grid state at election time.
    For performance, `update_global_color_distribution()` may avoid scanning the entire grid when areas are disjoint
    by aggregating cached per-area color counts plus a cached contribution from uncovered (static) cells.

- **Metrics**:
  - Participation rates, altruism factors, and metrics such as the Gini Index to analyze inequalities and long-term trends.

Learn more in the following sections.

Additional deep-dives:
- `docs/technical/semantics_representation_rng.md` (step semantics, vector contracts, RNG reproducibility policy)
- `docs/technical/voting_rules.md` (rule contracts, tie fairness, no-participation behavior)
- `docs/technical/core_mechanics.md` (distance coupling, fees/rewards algebra, break-even semantics)
- `docs/technical/population_preferences.md` (population composition, information level, and static preference-shape knobs)
- `docs/technical/participation_learning.md` (equations + implementation contract for participation learning)
- `docs/technical/altruism_learning.md` (equations + implementation contract for altruism learning)
- `docs/technical/run_control_output.md` (reproducibility, run control knobs, output contract)
- `docs/technical/decision_log.md` (risk-relevant semantic decisions and rationale)
