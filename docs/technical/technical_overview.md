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

---

### Features
- **Agents**:
  - Independently acting entities modeled with preferences, budgets, and decision-making strategies.
  - Can participate in elections, have personal preferences and limited information about surroundings.
  - Trained with decision-tree methods to simulate behavior.

- **Environment**:
  - Structured as a grid divided into "territories" or "areas."
  - A single unit of the grid is a "cell" or "field."
  - Each cell has a specific "color" representing a state. Elections influence these states, and areas mutate over time.

- **Metrics**:
  - Participation rates, altruism factors, and metrics such as the Gini Index to analyze inequalities and long-term trends.

Learn more in the following sections.