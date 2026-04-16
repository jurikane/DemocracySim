# Semantics, Representations & RNG

Several parts of DemocracySim only make sense if the underlying conventions are
kept fixed: what a recorded step means, how orderings and score vectors are
represented, and how seeded randomness is split across the simulation. The
points below are the compact description for those conventions.

## Step Semantics

Recorded step `t` is the election-time state. In other words, the system logs
the state on which the election at step `t` is evaluated, not a later
post-mutation state.

The execution order is:

1. Scheduler advances to `t`.
2. Mutation from election `t-1` is applied (for `t > 1`).
3. Elections/rewards/learning for `t` run.
4. Step-level data is collected and logged.

## Units

The main plotted summary series use these units:

- Turnout is stored in percent (`0..100`).
- Gini-based inequality metrics are stored on a percent-like scale (`0..100`).

## Representation Conventions

The main internal object types are:

- `Ordering`: permutation of option IDs (`0..n-1`).
- `Distribution`: non-negative vector summing to 1.
- `ScoreVector`: finite 1D vector in `[0,1]`.

## Tie Handling

Decision-critical tie breaks use seeded RNG. For a fixed seed, the same tie
cases reproduce the same outcomes.

## RNG Policy

- `run_seed = base_seed + run_id`.
- Core simulation RNG is deterministic for fixed config/seed.
- Participation and voting paths are isolated to avoid accidental coupling.
