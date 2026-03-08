# Population & Preferences

This page describes how agents and preference structures are initialized.

## Runtime Locations

- `src/models/participation_model.py::ParticipationModel.__init__`
- `src/models/participation_model.py::ParticipationModel._initialize_voting_agents`
- `src/agents/vote_agent.py::VoteAgent.__init__`
- `src/agents/vote_agent.py::VoteAgent._init_personal_opt_dist`
- `src/agents/vote_agent.py::VoteAgent.update_known_cells`

## Main Knobs

- `num_agents` (`>= 1`): number of voting agents.
- `initial_agent_assets` (`>= 0`): identical starting assets for all agents.
- `known_cells` (`>= 0`): number of cells sampled per step for local estimation.
- `num_personality_groups` (`>= 1`): number of group orderings (bounded by `factorial(num_colors)`).
- `personal_preference_peakedness` (`> 0`): concentration of `personal_opt_dist`.

## Semantics

- Personality groups are permutations over color IDs.
- `personal_opt_dist` is generated to follow each agent’s personality ordering.
- Knowledge sampling (`known_cells`) determines estimate confidence used in voting.
