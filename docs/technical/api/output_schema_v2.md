# Output schema v2

**Schema name:** `output_schema_v2`  
**Schema version:** `2`  
**Step indexing meaning:** `post_election_pre_mutation`

This document is the human-readable contract for the on-disk outputs produced by headless batch runs.

## Run directory layout

Per run directory (e.g. `.../data/simulation_output/<ts>/run_<i>/`):

- `meta.yaml` – config + seed + schema metadata
- `static.json` – static model info (grid size, personality_groups, file patterns)
- `steps.parquet`
- `area_steps.parquet`
- `agents.parquet`
- `votes.parquet`
- `grids/grid_0000.npy` (optional, **pre-election** snapshot for UI/replay convenience)
- `grids/grid_0001.npy` … `grids/grid_{S-1}.npy` (per-step grid snapshots)
- static overlays: `area_borders.npy`, `agents_per_cell.npy`, `area_strings_per_cell.npy`, `agent_strings_per_cell.npy`

## Timing semantics (important)

All Parquet tables are indexed as **post election + post reward, pre mutation**.

Meaning for step `t` (where **t starts at 1**):
- The election in each area has been conducted.
- Rewards and participation costs have been applied.
- **No color-cell mutation has been applied yet.**
- `area_color_*` in `area_steps.parquet` is the *real* area color distribution used at election time and by the reward logic.

### Replay step 0

Replay starts in a **grid-only step 0** state:
- It loads `grids/grid_0000.npy` (if present) and shows it as the initial grid.
- It does **not** populate model/area time series until the first replay `step()` call.

## Shared identifiers

All Parquet tables include:
- `run_seed` (int32): concrete RNG seed used for the run
- `rule_idx` (int16): voting rule index used for the run

## Tables

### `steps.parquet`

**Primary key:** `(run_seed, rule_idx, step)`

| column               |   dtype | notes                      |
|----------------------|--------:|----------------------------|
| run_seed             |   int32 | run identifier (seed)      |
| rule_idx             |   int16 | voting rule index          |
| step                 |   int32 | **1..S**                   |
| collective_assets    |   int64 | model sum of assets        |
| gini_index           |   int16 | 0–100                      |
| turnout              | float32 | global average turnout (%) |
| color_0..color_{C-1} | float32 | optional, C = num_colors   |

### `area_steps.parquet`

Merged area-state + election table.

**Primary key:** `(run_seed, rule_idx, step, area_id)`

| column                               |   dtype | notes                                 |
|--------------------------------------|--------:|---------------------------------------|
| run_seed                             |   int32 |                                       |
| rule_idx                             |   int16 |                                       |
| step                                 |   int32 |                                       |
| area_id                              |   int32 |                                       |
| eligible_voters                      |   int32 | area.num_agents                       |
| participants                         |   int32 | number who voted                      |
| turnout                              | float32 | participants/eligible_voters * 100    |
| election_cost_rate                   | float32 | from config/model                     |
| fee_pool                             | float32 | matches simulation internal type      |
| winning_option_id                    |   int32 | option row index into `model.options` |
| elected_color_0..elected_color_{C-1} |   int16 | `Area.voted_ordering`                 |
| dist_to_reality                      | float32 | distance(real_order, voted_order)     |
| gini_index                           |   int16 | area gini 0–100                       |
| area_color_0..area_color_{C-1}       | float32 | **pre-mutation distribution**         |

### `agents.parquet`

Agent snapshot table (**agent state only**).

**Primary key:** `(run_seed, rule_idx, step, agent_id)`

| column                     |   dtype | notes                                        |
|----------------------------|--------:|----------------------------------------------|
| run_seed                   |   int32 |                                              |
| rule_idx                   |   int16 |                                              |
| step                       |   int32 |                                              |
| agent_id                   |   int32 |                                              |
| row                        |   int16 |                                              |
| col                        |   int16 |                                              |
| assets                     | float32 | matches simulation internal type             |
| num_elections_participated |   int32 | cumulative counter across all areas/steps    |
| personality_group_idx            |   int16 |                                              |

**Semantics:** the row for step `t` represents the agent’s final state after it
participated in all elections it was eligible for during step `t`.

### `votes.parquet`

Vote signal table (participants only). This is the single source of
**election-contextual** agent values (belief/confidence).

**Primary key:** `(run_seed, rule_idx, step, area_id, agent_id)`

| column                                   |   dtype | notes                                          |
|------------------------------------------|--------:|------------------------------------------------|
| run_seed                                 |   int32 |                                                |
| rule_idx                                 |   int16 |                                                |
| step                                     |   int32 |                                                |
| area_id                                  |   int32 | disambiguates overlapping areas                |
| agent_id                                 |   int32 |                                                |
| participated                             | boolean | always true (rows only for participants)       |
| confidence                               | float32 | agent confidence at vote time **in this area** |
| estim_dst_color_0..estim_dst_color_{C-1} | float32 | estimated area color distribution at vote time |
| rank_1_option_id                         |   Int32 | option row index into `model.options`          |
| rank_1_oppose_score                      | float32 | lower = better                                 |
| rank_2_option_id                         |   Int32 |                                                |
| rank_2_oppose_score                      | float32 |                                                |
| rank_3_option_id                         |   Int32 |                                                |
| rank_3_oppose_score                      | float32 |                                                |

**Notes:**
- `votes.parquet` uses a fixed 3-rank wide layout to reduce row counts.
- If fewer than 3 options exist, remaining rank_* fields should be null.

## Notes

- Vector columns are **expanded**: `*_0..*_{C-1}` where `C=num_colors`.
- Validators live in `src/logging/output_schema_v2.py` and allow safe dtype upcasts.
