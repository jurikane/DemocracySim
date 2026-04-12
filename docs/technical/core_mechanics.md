# Core Mechanics: Distance, Fees, Rewards

This page summarizes the fee/reward pipeline used in elections.

## Key Quantities

- `fee = election_cost_rate * assets_pre` (participants only)
- `quality_distance`:
  - `dist_to_reality` if `quality_target_mode="reality"`
  - `puzzle_distance` if `quality_target_mode="puzzle"`
- `good_decision = (quality_distance <= break_even_distance_common)`
- `sign = +1` when good, `-1` when bad
- `group_dst_to_outcome = dist(preference_group, voted_ordering)`

Reward magnitude:

- if good: `factor = 1 - group_dst_to_outcome`
- if bad: `factor = group_dst_to_outcome`

Unified reward amount:

- `reward_amount = sign * reward_rate_personal * factor * assets_pre`

Per-election asset delta:

- `raw_delta_abs = reward_amount - fee`
- `assets_post = max(0, assets_pre + raw_delta_abs)`
- `delta_abs = assets_post - assets_pre`
- `delta_rel = delta_abs / assets_pre` if `assets_pre > 0`, else `0`

Participation learning consumes `delta_rel`.

## Runtime Locations

- Vote scoring and distances: `src/agents/strategies.py`, `src/utils/distance_functions.py`
- Vote tally: `src/agents/area.py::Area._tally_votes`
- Reward distribution: `src/agents/area.py::Area._distribute_rewards`
- Per-agent asset update: `src/agents/vote_agent.py::VoteAgent.reward_agent`

## Main Knobs

- `distance_idx`
- `election_cost_rate`
- `reward_rate_personal`
- `break_even_distance_common`
- `quality_target_mode`
- `puzzle_local_kappa`
- `puzzle_shock_prob`

## No-Participation Case

If no one participates in an area election, no new aggregate vote profile is computed for that step.
