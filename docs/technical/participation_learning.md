# Participation Learning

In DemocracySim, agents do not learn strategic ballot manipulation. They learn
whether to participate in later elections.

## Position In The Election Loop

Participation learning happens after an election outcome has been translated
into fees, rewards, and per-agent relative asset changes. The voting rule
therefore affects participation only indirectly: through the outcomes it
produces and the incentives that follow from them.

For the election pipeline around this update, see
[core_mechanics.md](core_mechanics.md).

## Decision Policy

Each agent carries a participation state `q_participation`. Participation
probability is logistic in that state:

- `p = sigmoid(participation_beta * q_participation)`

The actual decision is then sampled from a Bernoulli draw with probability `p`.

## Current Default Signal Mode

The default mode used in the thesis-oriented configuration is:

- `participation_signal_mode = group_relative_delta_rel_party`

Its logic is group-relative rather than purely individual. For each preference
group `g`, the model computes the mean relative asset change of the eligible
agents in that group:

- `mu_g = mean(delta_rel of eligible agents in group g)`

These group means are centered against the mean across groups:

- `mu_groups = mean(mu_g across groups)`
- `group_component_g = (mu_g - mu_groups) * n_g / (n_g + participation_signal_group_shrink_k)`

For an individual agent `i`, the signal then combines that group-relative term
with fee salience:

- `fee_component_i = - participation_signal_fee_weight * fee_rel_i` for participants, else `0`
- `signal_i = clip(group_component_g + fee_component_i, ±participation_signal_clip)`
- `q_push_i = signal_i`

The update is:

- `q_participation <- q_participation + participation_alpha * q_push_i`

If configured, `q_participation` is clipped symmetrically by
`participation_q_max`.

## Why This Signal Is Structured This Way

The model includes a costly participation decision and collective outcome
effects. Rewards and penalties are shared within groups, while the
participation fee is paid only by participants. That creates a free-rider
structure.

The default signal mode therefore does not treat participation learning as pure
individual reward chasing. It frames the update as a combination of:

- how an agent's preference group performed relative to other groups
- how much participation cost the agent personally had

## Main Knobs

The main parameters for participation learning are:

- `participation_alpha`
- `participation_beta`
- `participation_init_q`
- `participation_q_max`
- `participation_signal_mode`
- `participation_signal_fee_weight`
- `participation_signal_group_shrink_k`
- `participation_signal_clip`
- `participation_baseline_alpha`

## Implementation Reference

The group-level signal construction lives with the election logic in
[Area](api/Area.md). Per-agent q-state updates live in
[VoteAgent](api/VoteAgent.md), and the participation decision policy is exposed
in [Strategies](api/Strategies.md).

## Compatibility Notes

Other signal modes still exist in code for compatibility and comparative
experiments, including `raw_delta_rel` and
`group_centered_delta_rel_plus_fee`. The main current path is
`group_relative_delta_rel_party`.
