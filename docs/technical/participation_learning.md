# Participation Learning

This page documents how participation propensity is updated.

## Runtime Flow

Per step:

1. Agents decide whether to participate.
2. Election is aggregated.
3. Fee/reward outcomes are applied (`delta_rel` is computed per eligible agent).
4. Participation signal is built.
5. `q_participation` is updated.

Runtime locations:

- `src/agents/area.py::Area._compute_participation_learning_signals_and_q_pushes`
- `src/agents/vote_agent.py::VoteAgent.apply_participation_q_push`
- `src/agents/strategies.py::DefaultParticipationStrategy.decide_participation`

## Decision Policy

Participation probability is logistic in `q_participation`:

- `p = sigmoid(participation_beta * q_participation)`
- action is sampled from `Bernoulli(p)`

## Active Signal Mode (Thesis Baseline)

`participation_signal_mode = group_relative_delta_rel_party`

For each preference group `g`:

- `mu_g = mean(delta_rel of eligible agents in group g)`
- `mu_groups = mean(mu_g across groups)`
- `group_component_g = (mu_g - mu_groups) * n_g / (n_g + participation_signal_group_shrink_k)`

Per agent `i`:

- `fee_component_i = - participation_signal_fee_weight * fee_rel_i` for participants, else `0`
- `signal_i = clip(group_component_g + fee_component_i, ±participation_signal_clip)`
- `q_push_i = signal_i`

Update:

- `q_participation <- q_participation + participation_alpha * q_push_i`
- optional symmetric clipping by `participation_q_max`

## Key Knobs

- `participation_alpha`
- `participation_beta`
- `participation_init_q`
- `participation_q_max`
- `participation_signal_mode`
- `participation_signal_fee_weight`
- `participation_signal_group_shrink_k`
- `participation_signal_clip`
- `participation_baseline_alpha` (diagnostic baseline tracking)

## Compatibility

Other signal modes exist in code (`raw_delta_rel`, `group_centered_delta_rel_plus_fee`) for compatibility and comparative experiments.
