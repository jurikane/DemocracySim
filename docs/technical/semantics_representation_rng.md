# Semantics, Representations & RNG

This document holds the contracts for:
step semantics, representation safety, and reproducibility policy.

## Runtime Semantics (Authoritative)

Recorded step `t` represents the **election-time state**:

- mutation from election `t-1` is applied at start of step `t`
- elections/rewards/learning run on that state
- mutation from election `t` is applied at start of step `t+1`

Consequences:

- logged `steps.parquet` / `area_steps.parquet` are election-time aligned
- grid snapshots and table series must refer to the same step meaning

Turnout and Gini unit policy:

- turnout is stored in percent (`0..100`)
- gini is stored on a percent-like scale (`0..100`)

## Representation Contracts

Core vector concepts:

- `Ordering`: permutation of option ids (`0..n-1`)
- `Distribution`: non-negative vector summing to 1
- `ScoreVector`: finite 1D vector in `[0,1]` (thesis contract)

Fail-loud policy:

- invalid vectors must raise at representation entry points
- malformed vote vectors are not silently accepted

Tie policy:

- deterministic id-based tie-breaking is biased in decision-critical paths
- decision-critical conversions require explicit RNG on ties
- if tie-breaking happens without RNG in non-critical helpers, it is warning-visible

## RNG Determinism Policy

- Run seed derivation is deterministic (`run_seed = base_seed + run_id`)
- Simulation RNG stream is isolated from visualization/debug streams
- same config + same seed must reproduce identical core outputs

## Why This Matters for Thesis Validity

- Silent pre/post-mutation drift can invalidate time-series interpretation.
- Ambiguous vector semantics can produce plausible but wrong elections/rewards.
- Non-isolated RNG usage can break reproducibility without obvious errors.

## Test Coverage (What Is Locked By Pytests)

Semantics and units:

- `tests/test_step_semantics_mutation_timing.py`
- `tests/test_turnout_units_schema_v2.py`
- `tests/test_live_headless_replay_equivalence.py`

Representation contracts:

- `tests/test_representation_entrypoints_fuzz.py`
- `tests/test_representation_conversions.py`
- `tests/test_representations_contract.py`
- `tests/test_representation_contracts_core.py`

RNG and tie behavior:

- `tests/test_headless_determinism.py`
- `tests/test_rng_stream_isolation.py`
- `tests/test_tie_break_fairness.py`
- `tests/test_rule_tie_seed_contract.py`
