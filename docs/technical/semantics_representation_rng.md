# Semantics, Representations & RNG

This page defines step semantics and reproducibility conventions.

## Step Semantics

Recorded step `t` is the election-time state.

Execution order:

1. Scheduler advances to `t`.
2. Mutation from election `t-1` is applied (for `t > 1`).
3. Elections/rewards/learning for `t` run.
4. Step-level data is collected and logged.

## Units

- Turnout is stored in percent (`0..100`).
- Gini-based inequality metrics are stored on a percent-like scale (`0..100`).

## Representation Conventions

- `Ordering`: permutation of option IDs (`0..n-1`).
- `Distribution`: non-negative vector summing to 1.
- `ScoreVector`: finite 1D vector in `[0,1]`.

## Tie Handling

- Decision-critical tie breaks use seeded RNG.
- Same seed reproduces the same tie outcomes.

## RNG Policy

- `run_seed = base_seed + run_id`.
- Core simulation RNG is deterministic for fixed config/seed.
- Participation and voting paths are isolated to avoid accidental coupling.
