# How do different voting rules influence the temporal evolution of participation rates and inequality in a simple multi-agent system with adaptive agents?

This thesis investigates how different voting rules influence the temporal evolution of participation rates
and inequality in a simple multi-agent simulation with adaptive agents.

Agents are heterogeneous in preferences and repeatedly decide whether to participate in area-level elections
that aggregate individual preferences into collective decisions with payoff consequences.
Agents may start with equal resources, but resource inequality can emerge endogenously through repeated relative
reward/cost updates and adaptive behavior.
The matching or mismatching of agents’ preferences with their environment (and/or their perception of it) yields a time-varying measure of satisfaction/dissatisfaction.

The environment evolves under stationary update rules in response to collective decisions, which shape future reward distributions through preference matching or mismatching across agents. Elections therefore exert both immediate payoff effects and lagged effects via environmental change.

Agent behavior adapts via a fixed, explicit learning mechanism based on experienced outcomes, ensuring non-random, time-dependent dynamics.

The study compares a small set of canonical voting rules (2–4) while keeping all other model components fixed.
For final experiments, the voting rule is the only intentionally varied independent variable.

Outcomes are evaluated as time-series and summary statistics, with primary focus on participation and inequality:

- Participation dynamics: turnout over time.
- Inequality dynamics (resource dimension): Gini over agent assets (`gini_assets`).
- Inequality dynamics (experiential dimension): Gini over agent dissatisfaction (`gini_dissatisfaction`), where dissatisfaction is operationalized by `satisfaction_value` (distribution mismatch distance).

Important semantic clarification for interpretation:
`assets` are modeled as a generic resource/capacity state under relative reward/fee updates, not literal currency.
Accordingly, `gini_assets` is interpreted as inequality in simulation resource capacity.

The thesis explicitly excludes strategic voting, complex learning models, empirical validation, policy recommendations, and normative notions such as optimal democratic design.

Execution-level implementation scope and experiment triage are frozen in:
`docs/research/execution_scope_freeze.md`.
