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

Experiments vary `rule_idx` (the voting rule) while keeping all other components fixed.
Current implemented rules:

- `majority_rule` (plurality/first-choice)
- `approval_voting`
- `utilitarian_rule` (minimize total disagreement)
- `borda_rule` (positional scoring derived from per-voter orderings)

---

## Distance Functions (Fixed for Thesis Runs)

The model uses an ordering distance `distance_idx` for:

- ballot scoring (distance between the agent’s target ordering and each option ordering)
- reward signals (e.g. `dist_to_reality`)

Implemented distances (normalized to `[0,1]`):

- `spearman_fr_order` (Spearman footrule)
- `kendall_tau_order` (Kendall tau)

## Features

- **Agents**:
  - Independently acting entities modeled with preferences, budgets, and decision-making strategies.
  - Can participate in elections, have personal preferences and limited information about surroundings.
  - Trained with decision-tree methods to simulate behavior.
  - Participation costs are modeled via `election_cost_rate` (fraction of current assets paid when voting).
  - Rewards/penalties are scaled by wealth via `reward_rate_common` and `reward_rate_personal` (fractions of assets).
  - Reward signs/magnitudes are controlled by break-even points: `break_even_distance_common` 
    and `break_even_distance_personal`. These are *break-even distances*: coefficient = `break_even - distance`.
    If `distance < break_even`, the coefficient is positive (reward); if `distance > break_even`, negative (penalty).
  - Abstainers can receive a scaled share of the **common** reward/penalty via `abstention_share` (0..1). For abstainers:
    `common_component *= abstention_share` (participants always receive the full common component).

- **Environment**:
  - Structured as a grid divided into "territories" or "areas."
  - A single unit of the grid is a "cell" or "field."
  - Each cell has a specific "color" representing a state. Elections influence these states, and areas mutate over time.

- **Metrics**:
  - Participation rates, altruism factors, and metrics such as the Gini Index to analyze inequalities and long-term trends.

Learn more in the following sections.

Additional deep-dives:
- `docs/technical/participation_learning.md` (equations + implementation contract for participation learning)
