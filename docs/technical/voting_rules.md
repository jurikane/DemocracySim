# Voting Rules

This page documents the implemented voting rules and their runtime behavior.

## Implemented Rules

- `majority_rule`
- `approval_voting`
- `utilitarian_rule`
- `borda_rule`
- `schulze_rule`
- `random_rule`

Rule indices (`rule_idx`) used in config/runtime:

- `0` majority
- `1` approval
- `2` utilitarian
- `3` borda
- `4` schulze
- `5` random

Optional variant:

- `approval_voting_custom`

## Rule Input and Output

Input:

- Preference score table (`agents x options`, lower disagreement is better).

Output:

- Social ordering of options (best first).

## No-Participation Case

If no agents participate in an area election, no new aggregate ranking is computed for that election.

## Approval Mapping

- `approval_voting` uses fixed disagreement threshold `tau=0.5`.
- Ties in approval counts are resolved by:
  1. lower aggregate disagreement
  2. seeded RNG tie-break

## Tie Handling

- Ties are resolved with seeded RNG in decision-critical rule paths.
- Same seed yields reproducible outcomes.

## Schulze Notes

- `schulze_rule` operates on option candidates (same candidate set as other rules).
- Pairwise comparisons use strict score inequality (`<`); exact equal scores are neutral.
- If candidate count exceeds `120` options, Schulze fails fast with a clear error.
