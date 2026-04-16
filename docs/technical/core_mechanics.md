# Core Mechanics: Election Loop, Quality, Fees, Rewards

This page describes what happens around an election step once agents have
entered the model. It focuses on the shared mechanics that surround the voting
rule: participation, quality evaluation, fees, rewards, and the no-participation
case.

For how ballots are aggregated into a collective ordering, see
[voting_rules.md](voting_rules.md). For the default outcome quality target, 
see [puzzle_quality_gate_concept.md](../research/puzzle_quality_gate_concept.md). 
For the later update of participation propensity, see [participation_learning.md](participation_learning.md).

## Election Step

Within an area, the election-time logic is:

1. Agents decide whether to participate.
2. Participants pay an election fee and submit ballots.
3. The chosen voting rule aggregates the ballots into a full
   collective ordering.
4. The top-ranked option becomes the elected color ordering.
5. Outcome quality is evaluated against the active quality target.
6. Rewards or penalties are applied to agents.
7. Participation learning uses the resulting signals.
8. Mutation influenced by the elected ordering is applied at the start of the
   next scheduler step.

This means elections are evaluated on the current election-time state, while
grid mutation is lagged to the following step.

## Quality, Fees, And Rewards

Participants pay an election fee proportional to their current assets:

- `fee = election_cost_rate * assets_pre`

Outcome quality is represented by `quality_distance`. Its source depends on the
active quality mode:

- `dist_to_reality` if `quality_target_mode="reality"`
- `puzzle_distance` if `quality_target_mode="puzzle"`

An election counts as good if:

- `quality_distance <= break_even_distance_common`

Reward direction is then binary:

- good decision: positive sign
- bad decision: negative sign

Reward magnitude depends on how far a preference group is from the elected
ordering:

- `group_dst_to_outcome = dist(preference_group, voted_ordering)`

With `reward_rate_personal` as the common scale:

- if the decision is good, reward factor = `1 - group_dst_to_outcome`
- if the decision is bad, reward factor = `group_dst_to_outcome`

The resulting reward amount is:

- `reward_amount = sign * reward_rate_personal * factor * assets_pre`

After combining reward and fee, the per-election asset change is:

- `raw_delta_abs = reward_amount - fee`
- `assets_post = max(0, assets_pre + raw_delta_abs)`
- `delta_abs = assets_post - assets_pre`
- `delta_rel = delta_abs / assets_pre` if `assets_pre > 0`, else `0`

This `delta_rel` is one of the main inputs into participation learning.

## No-Participation Case

If no one participates in an area election, the model does not compute a new
aggregate ranking. Instead, the previous elected ordering remains in force. If
there is no previous elected ordering yet, the area falls back to the ordering
implied by the current realized area distribution.

Even in that case, the model still updates quality-distance monitoring and
group-distance tracking. What is missing is the new collective choice, the fee
and reward cycle tied to new participation, and the usual participation update
that depends on those election outcomes.

## Main Knobs

The main parameters for this part of the model are:

- `distance_idx`
- `election_cost_rate`
- `reward_rate_personal`
- `break_even_distance_common`
- `quality_target_mode`
- `puzzle_local_kappa`
- `puzzle_shock_prob`

## Runtime Locations

- vote tally and no-participation handling: `src/agents/area.py::Area.conduct_election`
- vote collection: `src/agents/area.py::Area._tally_votes`
- reward distribution: `src/agents/area.py::Area._distribute_rewards`
- per-agent asset update: `src/agents/vote_agent.py::VoteAgent.reward_agent`
- quality distances: `src/agents/area.py::Area._update_quality_distances`
