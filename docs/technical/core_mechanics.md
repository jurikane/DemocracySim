# Core Mechanics: Distance, Fees, Rewards

This document is the single reference for section **C** contracts:
distance coupling, fee/reward algebra, and break-even semantics.

## Where It Happens (Runtime Path)

- vote scoring / ordering distance:
  - `src/agents/strategies.py`
  - `src/utils/distance_functions.py`
- election cost and vote tally:
  - `src/agents/area.py::Area._tally_votes`
- reward decomposition:
  - `src/agents/area.py::Area._distribute_rewards`
- per-agent delta application:
  - `src/agents/vote_agent.py::VoteAgent.reward_agent`

## Authoritative Algebra

Per participating agent:

- `fee = election_cost_rate * assets_pre`

Common reward component:

- `common_coeff = break_even_distance_common - dist_to_reality`
- `common_component = reward_rate_common * assets * common_coeff`
- if abstaining: `common_component *= abstention_share`

`dist_to_reality` contract:

- uses tie-aware conversion of real color distributions
- ties are resolved by:
  - reference to `voted_ordering` when available (deterministic, no reward-path RNG)
  - unbiased RNG tie-break only when no reference ordering exists
- avoids option-id bias in tie handling

Personal reward component:

- `pers_coeff = break_even_distance_personal - personality_distance`
- `personal_component = reward_rate_personal * assets * pers_coeff`

Per-election raw absolute delta:

- `raw_delta_abs = common_component + personal_component - fee`

Relative learning signal input:

- `assets_post = max(0.0, assets_pre + raw_delta_abs)`
- `delta_abs = assets_post - assets_pre` (realized absolute delta)
- `delta_rel = delta_abs / assets_pre` (if `assets_pre > 0`, else `0.0`)

Asset update order (as implemented):

1. compute `raw_delta_abs` from reward components and fee
2. compute `assets_post = max(0.0, assets_pre + raw_delta_abs)`
3. compute realized `delta_abs = assets_post - assets_pre`
4. compute `delta_rel` from realized `delta_abs` and `assets_pre`
5. append realized `delta_abs` to `award_history`, set `assets = assets_post`

Learning consumption contract:

- participation learning consumes realized `delta_rel`

## Knob Semantics

- `distance_idx`: distance function used in election/reward paths
- `election_cost_rate`: participation fee rate (fraction of current assets)
- `reward_rate_common`: scaling rate for common component
- `reward_rate_personal`: scaling rate for personal component
- `break_even_distance_common`: sign pivot for common component
- `break_even_distance_personal`: sign pivot for personal component
- `abstention_share`: fraction of common component paid to abstainers

Interpretation of break-even distances:

- distance `< break_even` => positive coefficient (reward)
- distance `> break_even` => negative coefficient (penalty)

## No-Participation Semantics

- if an area has no participants, no new aggregate profile is computed
- no standard reward distribution is executed for that election
- area diagnostics still update `dist_to_reality` for monitoring

## Risk Addressed by Contracts

- Guardrails protect turnout/inequality dynamics from arithmetic drift in the fee/reward path.
- Break-even semantics are locked so adaptation does not silently flip from reward- to penalty-dominated behavior.
- Logging consistency checks prevent analysis from using plausible-looking but semantically wrong fee/reward traces.

## Test Coverage (What Is Locked By Pytests)

Distance and normalization:

- `tests/test_distance_idx_contract.py`
- `tests/test_distance_function_semantics_contract.py`

Fee and reward knobs:

- `tests/test_election_cost_rate_contract.py`
- `tests/test_reward_rate_common_contract.py`
- `tests/test_reward_rate_personal_contract.py`
- `tests/test_break_even_distance_common_contract.py`
- `tests/test_break_even_distance_personal_strong_contract.py`
- `tests/test_abstention_share_contract.py`

Pipeline visibility / fail-loud checks:

- `tests/test_output_pipeline_contract.py`
- `tests/test_relative_delta_signal_computed_pre_asset_update.py`

## Risk-Flag Rule (Development Policy)

For core dynamics code (fees/rewards/learning signals), any new:

- threshold (`if x < c`)
- clamp/floor/ceiling (`max/min/clip`)
- fallback normalization rule

must be treated as a **risk flag** and requires:

- explicit rationale in code/docs
- a boundary test around the threshold
- a property test for invariance/continuity where applicable
