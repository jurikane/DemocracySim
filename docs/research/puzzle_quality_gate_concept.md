# Puzzle Quality Gate Concept

This defines the conceptual and technical role of the puzzle process used for decision-quality evaluation.

Canonical term in thesis text: **Puzzle Quality Gate**.
Allowed shorthand in prose after definition: **puzzle**.
Avoid rotating through alternative labels in main thesis text.

## 1. Why the Puzzle Quality Gate Exists

The thesis requires a quality-pressure mechanism: decisions should be rewardable or punishable based on alignment with the current puzzle.

Prior to the puzzle, the quality reference was coupled to current grid state.
However, this created lock-in states by favoring alignment with start conditions and majority, that are not very interesting to study.
Now it is separated into its own stochastic process.

Key reason for the split:

- avoid fixed directional lock-in in quality pressure
- keep quality pressure symmetric over the simplex in expectation
- allow the realized grid state and the quality-pressure mechanism to diverge

## 2. Implemented Process

In puzzle mode (`quality_target_mode=puzzle`), each area has a puzzle distribution over colors.

Update logic per step:

1. If no previous puzzle exists: sample fresh Dirichlet distribution.
2. Otherwise:
   - with probability `puzzle_shock_prob`, redraw fresh Dirichlet sample
   - else sample local Dirichlet step around previous puzzle with concentration `puzzle_local_kappa`

Process properties:

- color-symmetric priors (no label favoritism)
- local persistence controlled by `puzzle_local_kappa`
- occasional jumps controlled by `puzzle_shock_prob`

## 3. Quality Gate Mechanics

After election, the elected ordering is compared to:

- puzzle ordering in puzzle mode (`puzzle_distance`)
- grid ordering in reality mode (`dist_to_reality`)

Decision sign is binary via threshold `break_even_distance_common`:

- if distance <= threshold: "good" mode (positive reward sign)
- else: "bad" mode (negative reward sign)

So the puzzle controls the sign regime, not just a descriptive metric.

## 4. Conceptual Interpretation

Puzzle Quality Gate represents changing external viability pressure that collective decisions need to track to avoid systemic penalties.

This does not mean the process is ontologically "true reality".
It is a modeled quality constraint process with controlled stochastic dynamics.

## 5. Why It Is Useful for This Thesis

The split enables analysis of three distinct forces:

- path-dependent collective self-shaping (grid)
- external quality pressure (Puzzle Quality Gate)
- aggregation structure (voting rule)

This makes it possible to study whether some voting rules better balance self-regarding pressure and quality alignment.

## 6. Limits and Non-Claims

The Puzzle Quality Gate is a stylized process and not an empirical exogenous data process.
It supports internal comparative validity, not external causal claims about real societies.
