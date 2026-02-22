# Core Mechanics: Distance, Fees, Rewards

This document is the single reference for section **C** contracts:
distance coupling, fee/reward algebra, and break-even semantics.

## Where It Happens (Runtime Path)

- vote scoring / ordering distance:
  - `src/agents/strategies.py`
  - `src/utils/distance_functions.py`
- election cost and vote tally:
  - `src/agents/area.py::Area._tally_votes`
- reward distribution:
  - `src/agents/area.py::Area._distribute_rewards`
- per-agent delta application:
  - `src/agents/vote_agent.py::VoteAgent.reward_agent`

## Authoritative Algebra (Binary Quality-Sign Reward)

Per eligible agent:

- `fee = election_cost_rate * assets_pre` (participants only, applied in `reward_agent`)

Quality gate:

- quality distance selection:
  - if `quality_target_mode == "reality"`: `quality_distance = dist_to_reality`
  - if `quality_target_mode == "puzzle"`: `quality_distance = puzzle_distance`
- `good_decision = (quality_distance <= break_even_distance_common)`
- `sign = +1 if good_decision else -1`

Group distance to election outcome:

- `group_dst_to_outcome = dist(personality_group, voted_ordering)` in `[0,1]`

Reward factor:

- if `good_decision`: `factor = 1 - group_dst_to_outcome`
- else: `factor = group_dst_to_outcome`

Unified reward amount:

- `reward_amount = sign * reward_rate_personal * factor * assets_pre`

Per-election raw absolute delta:

- `raw_delta_abs = reward_amount - fee`

Relative learning signal input:

- `assets_post = max(0.0, assets_pre + raw_delta_abs)`
- `delta_abs = assets_post - assets_pre` (realized absolute delta)
- `delta_rel = delta_abs / assets_pre` (if `assets_pre > 0`, else `0.0`)

Asset update order (as implemented):

1. compute `raw_delta_abs` from reward amount and fee
2. compute `assets_post = max(0.0, assets_pre + raw_delta_abs)`
3. compute realized `delta_abs = assets_post - assets_pre`
4. compute `delta_rel` from realized `delta_abs` and `assets_pre`
5. append realized `delta_abs` to `award_history`, set `assets = assets_post`

Learning consumption contract:

- participation learning consumes realized `delta_rel`

`dist_to_reality` contract:

- uses tie-aware conversion of real color distributions
- ties are resolved by:
  - reference to `voted_ordering` when available (deterministic, no reward-path RNG)
  - unbiased RNG tie-break only when no reference ordering exists
- avoids option-id bias in tie handling

## Knob Semantics

- `distance_idx`: distance function used in election/reward paths
- `election_cost_rate`: participation fee rate (fraction of current assets)
- `reward_rate_personal`: unified reward/punishment rate
- `break_even_distance_common`: quality threshold for sign switch
- `quality_target_mode`: source of quality distance (`reality` | `puzzle`)
- `puzzle_local_kappa`: local puzzle random-walk concentration (`>0`; higher means smaller jumps)
- `puzzle_shock_prob`: probability of a full puzzle redraw (rare large jumps)

## No-Participation Semantics

- if an area has no participants, no new aggregate profile is computed
- no standard reward distribution is executed for that election
- area diagnostics still update `dist_to_reality` for monitoring
- in puzzle mode, `puzzle_distance` is also updated for monitoring

## Risk Addressed by Contracts

- Guardrails protect turnout/inequality dynamics from arithmetic drift in the fee/reward path.
- Quality-sign semantics are explicit and test-locked.
- Logging consistency checks prevent analysis from using plausible-looking but semantically wrong fee/reward traces.

## Test Coverage (What Is Locked By Pytests)

Distance and normalization:

- `tests/test_distance_idx_contract.py`
- `tests/test_distance_function_semantics_contract.py`

Fee/reward pipeline:

- `tests/test_election_cost_rate_contract.py`
- `tests/test_reward_binary_quality_contract.py`
- `tests/test_removed_reward_knobs_contract.py`
- `tests/test_relative_delta_signal_computed_pre_asset_update.py`
- `tests/test_cp17_agent_causal_logging.py`

## Risk-Flag Rule (Development Policy)

For core dynamics code (fees/rewards/learning signals), any new:

- threshold (`if x < c`)
- clamp/floor/ceiling (`max/min/clip`)
- fallback normalization rule

must be treated as a **risk flag** and requires:

- explicit rationale in code/docs
- a boundary test around the threshold
- a property test for invariance/continuity where applicable
