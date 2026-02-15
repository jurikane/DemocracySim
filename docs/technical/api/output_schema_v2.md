# Output schema v2

**Schema name:** `output_schema_v2`  
**Schema version:** `2`  
**Step indexing meaning:** `post_election_pre_mutation`

This document is the human-readable contract for the on-disk outputs produced by headless batch runs.

## Run directory layout

Per run directory (e.g. `.../data/simulation_output/<ts>/run_<i>/`):

- `meta.yaml` – schema + run metadata, plus config reference
- `config_used.yaml` – canonical batch config (stored at batch root; `meta.yaml` points to it via `config_ref`)
- `static.json` – static model info (grid size, personality_groups, file patterns)
- `steps.parquet`
- `area_steps.parquet`
- `agents.parquet`
- `votes.parquet`
- `grids/grid_0000.npy` (optional, **pre-election** snapshot for UI/replay convenience)
- `grids/grid_0001.npy` … `grids/grid_{S}.npy` (optional per-step snapshots, **pre-mutation**, depending on grid_interval).
  `grid_0001.npy` may equal `grid_0000.npy` because no mutation is applied before step 1.
- static overlays: `area_borders.npy`, `agents_per_cell.npy`, `area_strings_per_cell.npy`, `agent_strings_per_cell.npy`

## Voting rule identification (reproducibility)

The *primary independent variable* for the thesis experiments is `rule_idx`.
To make `rule_idx` unambiguous across code changes, schema v2 stores:

- `meta.yaml`: `run.rule_idx`, `run.rule_name`, `run.rule_impl_name`
- `static.json`: `voting_rules.names`, `voting_rules.impl_names`, plus the selected index/name

The distance function (`distance_idx`) is also recorded for auditability:

- `meta.yaml`: `run.distance_idx`, `run.distance_name`, `run.distance_impl_name`
- `static.json`: `distance_functions.names`, `distance_functions.impl_names`, plus the selected index/name

## Timing semantics (important)

Recorded step `t` (where **t starts at 1**) corresponds to the election-time state:

- The grid shown/used is the state after applying mutation from step `t-1` (for `t>1`).
- Elections and rewards/learning happen on this state during step `t`.
- No mutation occurs during step `t`; mutation from step `t` is applied at the start of step `t+1`.
- `area_color_*` in `area_steps.parquet` reflects the election-time distribution for step `t`.
- `color_*` in `steps.parquet` reflects the global election-time distribution for step `t`,
  computed from exact grid counts (not from averaging area distributions).
- Grid snapshots (`grids/grid_*.npy`) are the election-time state for step `t` and match the distributions.
- When snapshots are stored sparsely (`grid_interval > 1`), replay uses carry-forward semantics:
  for step `t` it shows the latest available `grid_k` with `k <= t`.

`post_election_pre_mutation` means: post-election/reward for step `t`, pre-mutation of step `t` (applied at step `t+1`).

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
| collective_assets    | float32 | model sum of assets        |
| gini_index           |   int16 | 0–100                      |
| turnout              | float32 | global average turnout (%) |
| mean_altruism        | float32 | mean altruism_factor       |
| mean_dissatisfaction    | float32 | mean dissatisfaction_value      |
| color_0..color_{C-1} | float32 | optional, pre-mutation     |

### `area_steps.parquet`

Merged area-state + election table.

**Primary key:** `(run_seed, rule_idx, step, area_id)`

| column                               |   dtype | notes                                          |
|--------------------------------------|--------:|------------------------------------------------|
| run_seed                             |   int32 |                                                |
| rule_idx                             |   int16 |                                                |
| step                                 |   int32 |                                                |
| area_id                              |   int32 |                                                |
| eligible_voters                      |   int32 | agents eligible for election in this area/step |
| participants                         |   int32 | number who voted                               |
| turnout                              | float32 | participants/area_num_agents * 100             |
| election_cost_rate                   | float32 | fraction of assets paid by participants (0..1) |
| fee_pool                             | float32 | matches simulation internal type               |
| winning_option_id                    |   int32 | option row index into `model.options`          |
| elected_color_0..elected_color_{C-1} |   int16 | `Area.voted_ordering`                          |
| dist_to_reality                      | float32 | distance(real_order, voted_order)              |
| gini_index                           |   int16 | area gini 0–100         |
| area_color_0..area_color_{C-1}       | float32 | **pre-mutation distribution**                  |

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
| personality_group_idx      |   int16 |                                              |
| eligible_for_election      | boolean | eligibility flag in this area/step election  |
| participating              | boolean | participation decision in this area/step     |
| election_fee               | float32 | charged fee in this area/step                |
| reward_common_component    | float32 | common reward/penalty component              |
| reward_personal_component  | float32 | personal reward/penalty component            |
| election_delta_abs         | float32 | realized absolute asset delta                |
| election_delta_rel         | float32 | realized relative asset delta                |
| participation_baseline     | float32 | EMA baseline for participation learning      |
| participation_signal       | float32 | baseline-corrected participation signal      |
| altruism_factor            | float32 | agent altruism_factor                        |
| dissatisfaction_value         | float32 | dissatisfaction (distance)                   |
| dissatisfaction_baseline      | float32 | EMA baseline for dissatisfaction             |
| dissatisfaction_signal        | float32 | baseline-corrected dissatisfaction signal    |

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
| participating                            | boolean | always true (rows only for participants)       |
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
- If estimate distributions are missing/invalid at vote time, `estim_dst_color_*` is written as `NaN` (fail-visible), not zeros.
- If no agent participates in a run/step, `votes.parquet` may be empty (`0` rows) but remains schema-valid with typed columns.

## Notes

- Vector columns are **expanded**: `*_0..*_{C-1}` where `C=num_colors`.
- Validators live in `src/logging/output_schema.py` and allow safe dtype upcasts.
- Validators reject unknown columns (strict schema lock). Only documented fields and documented vector prefixes are accepted.
- Missing required pre-mutation area snapshot fields fail loudly during logging (no silent fallback).
