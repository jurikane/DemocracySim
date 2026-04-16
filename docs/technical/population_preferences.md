# Population & Preferences

In this model, preference groups define ordinal color
rankings, while individual agents additionally carry their own preference
distribution over those rankings.

## Implementation Reference

Population initialization and model-level preference setup are documented in
[ParticipationModel](api/Model.md). Agent-level preference construction and
knowledge updates are documented in [VoteAgent](api/VoteAgent.md).

## Main Knobs

- `num_agents` (`>= 1`): number of voting agents
- `known_cells` (`>= 0`): number of cells sampled per step for local estimation
- `num_personality_groups` (`>= 1`): number of preference-group orderings (bounded by `factorial(num_colors)`)
- `personal_preference_peakedness` (`> 0`): concentration of `personal_opt_dist`
- `initial_agent_assets` (`>= 0`): identical starting assets for all agents


## Semantics

Preference groups are permutations over color IDs. They determine the order in
which colors are preferred, but not how strongly they are preferred.

`personal_opt_dist` is then generated to follow each agent’s preference-group
ordering while still allowing variation in preference intensity across agents.

Knowledge sampling via `known_cells` determines how much local information an
agent has when constructing her current estimate for voting.
