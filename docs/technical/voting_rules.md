# Voting Rules

This page documents the implemented voting rules and their runtime behavior.

## Implemented Rules

- `plurality_rule`
- `approval_voting`
- `utilitarian_rule`
- `borda_rule`
- `schulze_rule`
- `random_rule`

Rule indices (`rule_idx`) used in config/runtime:

- `0` plurality
- `1` approval
- `2` utilitarian
- `3` borda
- `4` schulze
- `5` random

Optional legacy variant:

- `approval_voting_custom`

## Rule Input and Output

Input:

- score table with shape `agents x options`
- each option is a complete color ordering
- values are disagreement scores
- lower disagreement is better

Output:

- full social ordering of options, best first
- the winning option is then interpreted as the elected color ordering

## Operational Behavior

### Utilitarian

- sums disagreement scores across participating agents for each option
- ranks options from lower to higher total disagreement
- uses seeded randomized tie-breaking on equal totals

### Borda

- converts each agent's score row into an ordering by ascending disagreement
- assigns Borda points from best to worst (`m-1` down to `0`)
- sums points across agents
- ranks options from higher to lower total Borda score
- uses seeded tie-breaking both in per-agent order construction and final ties

### Schulze

- operates on the same option set as the other rules
- builds pairwise comparisons from strict score inequality (`<`)
- exact equal scores are neutral
- applies the Schulze strongest-path method to obtain the collective ordering
- fails fast above `120` options for predictable runtime

### Plurality

- uses only first choices
- first-choice ties inside a ballot are broken before counting by seeded tie preparation
- counts first-choice votes to determine the plurality stage
- completes lower ranks afterward by seeded random completion so the output remains a full ordering

### Approval

- uses fixed disagreement threshold `tau = 0.5`
- an option is approved if its disagreement score is strictly below `tau`
- ranks options by approval count
- tie stack:
  1. lower aggregate disagreement
  2. seeded RNG tie-break

### Random

- ignores ballot content
- draws a full ordering uniformly at random
- deterministic for a fixed RNG seed

## Tie Handling and Reproducibility

- decision-critical randomized tie handling is seeded
- the same seed yields the same ballot preparation, tie resolution, and final ordering

## No-Participation Case

If no agents participate in an area election, no new aggregate ranking is computed for that election.
