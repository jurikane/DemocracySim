# Population & Preferences

This document is the single reference for **agent-population and preference semantics**:
how many agents are created, which initial resources they receive, how much local
information they have, and how static preference profiles are constructed.

## Where It Happens (Runtime Path)

Population and preference setup is fixed during model initialization:

1. `ParticipationModel.__init__` validates population/preference knobs.
2. `_initialize_voting_agents()` creates all agents and assigns initial assets.
3. `VoteAgent.__init__` initializes:
   - personality group assignment
   - `personal_opt_dist` from personality ordering + peakedness
4. During step execution, `VoteAgent.update_known_cells()` defines the
   sampled local knowledge used for vote estimation.

Code references:

- `src/models/participation_model.py::ParticipationModel.__init__`
- `src/models/participation_model.py::ParticipationModel._initialize_voting_agents`
- `src/agents/vote_agent.py::VoteAgent.__init__`
- `src/agents/vote_agent.py::VoteAgent._init_personal_opt_dist`
- `src/agents/vote_agent.py::VoteAgent.update_known_cells`

## Semantics (Authoritative)

### `num_agents`

- Integer `>= 1`.
- Exactly `num_agents` voting agents are initialized and logged.

### `initial_agent_assets`

- Finite `>= 0`.
- Every agent starts with the same initial asset level.

### `known_cells`

- Integer `>= 0`.
- Number of area cells each agent samples per step for local estimation.
- `0` is valid and yields zero confidence.

### `num_personality_groups`

- Integer `>= 1`.
- Must be `<= factorial(num_colors)` (max number of unique orderings).
- Personality groups are unique permutations over color IDs.

### `personal_preference_peakedness`

- Finite `> 0`.
- Controls how concentrated per-agent `personal_opt_dist` is:
  - `>1`: more peaked
  - `<1`: flatter
  - `=1`: neutral baseline

## Why This Matters for Thesis Validity

- These knobs define the social composition and information structure of the electorate.
- Silent errors here distort turnout dynamics, reward signals, and inequality trajectories.
- Fail-loud contracts ensure that changes in outcomes come from intended treatments, not hidden setup drift.

## Logging and Integration Expectations

- `agents.parquet` contains one row per agent per step with valid assets and participation fields.
- `votes.parquet` confidence reflects knowledge sampling (`known_cells`) under fixed topology.
- `static.json` stores `personal_opt_dist` and personality-group metadata consistent with configured knobs.

## Test Coverage (What Is Locked By Pytests)

Core contracts:

- `tests/test_num_agents_contract.py`
- `tests/test_initial_agent_assets_contract.py`
- `tests/test_known_cells_contract.py`
- `tests/test_num_personality_groups_contract.py`
- `tests/test_personal_preference_peakedness_contract.py`

Interaction test:

- `tests/test_population_preference_interactions.py`
  - exercises joint behavior of `known_cells`, `num_personality_groups`,
    `personal_preference_peakedness`, and `initial_agent_assets`
  - checks directional effects in logged outputs (`votes.parquet`, `agents.parquet`, `static.json`)

## Recommended Thesis Run Policy

For baseline experiment grids:

- keep population/preference knobs fixed across voting-rule comparisons
- use `personal_preference_peakedness = 1.0` as baseline unless explicitly studying preference intensity effects
- treat changes in information level (`known_cells`) as robustness analyses, not baseline variation
