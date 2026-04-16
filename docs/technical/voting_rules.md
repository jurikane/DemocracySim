# Voting Rules

DemocracySim aggregates ballots inside an election. The
rules differ only in how they turn the participating agents' ballots into a
collective ordering. Costs, rewards, participation learning, and later grid
mutation are part of the broader election loop and are documented in
[core_mechanics.md](core_mechanics.md) and
[participation_learning.md](participation_learning.md).

## Ballots And Option Space

In this model, an election option is not a single color but a complete ordering
of the colors. With four colors, the option set therefore consists of all
`4! = 24` possible color orderings.

Each participating agent submits a disagreement vector over these options.
Lower disagreement is better. A voting rule maps the profile of these vectors
to a full collective ordering of the options, and the top-ranked option is then
interpreted as the elected color ordering.

This representation matters for reading the rules below. The rules are not
choosing directly among colors, but among complete orderings of colors.







`approval` is kept as a context arm. It remains useful for comparison and
calibration, but it is not part of the canonical confirmatory family.

## Rule Mapping In Runtime

The runtime mapping used by `model.rule_idx` is:

- `0 = plurality`
- `1 = approval`
- `2 = utilitarian`
- `3 = borda`
- `4 = schulze`
- `5 = random`

For where this appears in configuration and stored run artifacts, see
[run_control_output.md](run_control_output.md).

## Rule Definitions

### Utilitarian

`utilitarian` aggregates ballots directly on the disagreement scores. For each
option, it sums disagreement across participating agents and ranks options from
lower to higher total disagreement. In this sense, it selects the option with
the smallest collective disagreement.

### Borda

`borda` first converts each ballot into an ordering of the options by ascending
disagreement. On a set of `m` options, the top-ranked option receives `m - 1`
points, the next `m - 2`, and so on down to `0`. These points are summed
across participating agents, and options are ordered from higher to lower total
Borda score.

### Schulze

`schulze` also works on induced rankings per ballot. For two options `a` and
`b`, an agent counts as preferring `a` to `b` if the disagreement score of `a`
is strictly lower than the disagreement score of `b`. Exact ties are neutral.
The resulting pairwise comparison matrix is then aggregated with the Schulze
strongest-path method to obtain a collective ordering.

Because the candidate set here is the full option space of color orderings,
Schulze is guarded against very large option counts for runtime reasons.

### Plurality

`plurality` uses only first choices. Before counting, ties among equally best
options within a ballot are broken with seeded tie preparation. The plurality
winner is then the option with the largest number of first-choice votes.

Because the model requires a full collective ordering rather than only a
winner, the lower ranks are completed after the plurality stage by seeded
random ordering of the remaining options.

### Approval

`approval` maps disagreement scores to binary approvals with a fixed threshold
`tau = 0.5`. An option is approved if its disagreement score is below that
threshold. Options are ordered first by approval count, then by lower total
disagreement, and only then by seeded random tie resolution.

### Random

`random` ignores ballot content and draws a full ordering uniformly at random
from the available options. Because the draw is seeded, the result is still
reproducible for a fixed run seed.

## No-Participation Case

If no agents participate in an election, no new aggregate ranking is computed.
The previous elected ordering remains in force. If the election has no previous
ordering yet, the current realized ordering of the area is used instead.

For the broader consequences of that case within the election pipeline, see
[core_mechanics.md](core_mechanics.md).

## Tie Handling And Reproducibility

Decision-relevant randomized tie handling is seeded. For a fixed seed, the same
run reproduces the same ballot preparation, tie resolution, and final ordering.
