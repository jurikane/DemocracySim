# Voting Rules & Rule Index Contract

This document is the single reference for section **B** contracts:
the primary independent variable (voting rule), rule semantics, and logging identity.

## Rule Set

Implemented rule set:

- `majority_rule`
- `approval_voting`
- `utilitarian_rule`
- `borda_rule`
- `random_rule` (reference arm)

Additional non-baseline variant available:

- `approval_voting_custom` (adaptive `mean-variance` threshold mapping)

For baseline thesis experiments, the voting rule is the only intentionally varied independent variable.
Canonical confirmatory reporting and random-reference reporting are separated in the thesis inference layer.

## Rule Input/Output Contract

Input:

- preference table as ScoreVectors (`rows=agents`, `cols=options`, lower=better)

Output:

- social Ordering of options (best first)

No-participation edge:

- if no agents participate in an area election, no new aggregate ranking is computed
- area keeps previous outcome (or initializes from current area reality ordering on first occurrence)
- this behavior is explicit and logged (not silently fabricated as a vote profile)

Approval mapping policy:

- `approval_voting` uses a fixed threshold on disagreement scores (`score <= tau`, with `tau=0.5`)
- ties on approval counts are resolved by:
  1. lower aggregate disagreement
  2. seeded RNG as final tie-break
- `approval_voting_custom` keeps the legacy adaptive threshold behavior for exploratory use only

## Rule Identity and Auditability

Run metadata stores rule identity in multiple forms:

- index
- short display name
- implementation name

This protects analysis from silent remapping or refactor drift.

## Tie Handling

- ties are resolved using explicit RNG in rule implementations
- same seed => deterministic replay of tied outcomes
- across many seeds, tied outcomes should not show systematic option-id bias
- `majority_rule` randomization is restricted to genuine first-choice ties;
  strict (non-tied) first choices are not perturbed by noise.

## Why This Matters for Thesis Validity

- The whole causal comparison hinges on rule identity correctness.
- Tie handling can silently bias rule comparisons if neutrality is not enforced.
- No-participation semantics affect reward and learning trajectories in sparse-turnout regimes.

## Test Coverage (What Is Locked By Pytests)

Rule correctness and semantics:

- `tests/test_majority_rule.py`
- `tests/test_approval_voting.py`
- `tests/test_voting_rules_additional.py`
- `tests/test_no_participation_debug_snapshot.py`

Rule identity and metadata:

- `tests/test_rule_idx_metadata_static_json.py`

Fairness / invariance / determinism:

- `tests/test_tie_break_fairness.py`
- `tests/test_rule_label_permutation_invariance.py`
- `tests/test_rule_tie_seed_contract.py`
- `tests/test_random_rule_contract.py`
