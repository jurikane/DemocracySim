# Altruism Learning (Technical Contract)

This document is the single reference for **how altruism learning works in code**:
what is updated, when it is updated, and why the model is designed this way.

## What “Altruism” Means in This Model

Each agent has an `altruism_factor` in `[0,1]` that controls **vote mode switching**:

- with probability `altruism_factor`: vote **altruistically** (reality-tracking)
- with probability `1 - altruism_factor`: vote **self-regardingly** (personality-ordering)

Default voting strategy:

- self-regarding mode: returns precomputed static oppose-scores from the agent’s `personality_group` ordering
- altruistic mode: estimates area distribution, converts to ordering, then scores options by ordering-distance

Code references:

- Default voting: `src/agents/strategies.py::DefaultVotingStrategy.score_options`
- Agent-held self-regarding scores: `src/agents/vote_agent.py::VoteAgent.self_regarding_oppose_scores`

## Where It Happens (Runtime Path)

Per model step (`t = 1,2,...`) the relevant path is:

1. `ParticipationModel.step()` calls `scheduler.step()`
2. `CustomScheduler.step()` calls `Area.step()` for each area
3. `Area.step()`:
   - updates each agent’s knowledge + satisfaction (pre-election)
   - calls `Area.conduct_election()` which runs elections + rewards
   - **then** optionally applies altruism learning updates (participant-only)

Code references:

- Dissatisfaction baseline/signal: `src/agents/area.py::Area.step`
- Participant-only altruism update: `src/agents/area.py::Area.conduct_election`
- Update rule: `src/agents/vote_agent.py::VoteAgent.apply_altruism_update`

## State Variables (Per Agent)

Altruism learning uses:

- `altruism_factor` (float): the agent’s current reality-weight
- `dissatisfaction_value` (float): dissatisfaction / distance to a target distribution (depends on `satisfaction_mode`)
- `dissatisfaction_baseline` (float): EMA baseline of dissatisfaction values, initialized as `NaN`
- `dissatisfaction_signal` (float): baseline-corrected dissatisfaction surprise signal (`dv - baseline`)

## Dissatisfaction Signal (Input to Altruism Learning)

Dissatisfaction is computed **before** the election in `Area.step()`:

1. compute `dv = agent.compute_dissatisfaction_value(area, model)`
2. baseline initialize on first observation:

    ```text
    baseline <- dv
    signal   <- 0
    ```

3. otherwise:

    ```text
    signal   <- dv - baseline
    baseline <- (1 - satisfaction_baseline_alpha) * baseline
              + satisfaction_baseline_alpha * dv
    ```

Interpretation:

- `signal > 0` means “worse than expected” (more dissatisfied than baseline)
- `signal < 0` means “better than expected” (less dissatisfied than baseline)

Note: in this project, dissatisfaction is a **distance** (bigger = worse), so the sign interpretation differs
from “reward signals”.

## Altruism Learning Toggle and Initialization

Model-level toggle:

- `altruism_learning` (bool)

Initialization:

- if `altruism_learning=True`, each agent starts with `altruism_factor = altruism_init`
- if `altruism_learning=False`, each agent uses a fixed `altruism_factor = altruism_static`

This lets you treat altruism either as:

- a fixed heterogeneity parameter (static), or
- an adaptive behavioral parameter (learning).

## Update Rule (Participant-Only, Baseline Thesis Mechanism)

Altruism learning is applied **only** to participating agents, after the election:

```text
if altruism_learning and participating:
    altruism_factor <- altruism_factor + altruism_alpha * dissatisfaction_signal
    altruism_factor <- clip(altruism_factor, [altruism_clip_min, altruism_clip_max])
```

Design choices:

- **Participant-only**: only those who “acted” (voted) adapt their reality-weight this step.
- **Linear update**: intentionally simple and auditable (fits thesis scope).
- **Clipping**: prevents runaway values and keeps the mode-probability interpretation well-defined.

Practical intuition:

- If dissatisfaction is higher than expected (`signal > 0`), altruism increases (higher probability of altruistic votes).
- If dissatisfaction is lower than expected (`signal < 0`), altruism decreases (higher probability of self-regarding votes).

Whether this produces stable dynamics depends on the dissatisfaction signal statistics and `altruism_alpha`.

## Knobs (What They Mean)

Altruism learning knobs (ModelConfig):

- `altruism_learning` (bool): learning on/off
- `altruism_static` (in `[0,1]`): fixed altruism when learning is off
- `altruism_init` (in `[0,1]`): initial altruism when learning is on
- `altruism_alpha` (>= 0): learning rate for altruism updates
- `altruism_clip_min`, `altruism_clip_max` (finite, `min <= max`): clip interval for altruism_factor

Dissatisfaction knobs (inputs to altruism updates):

- `satisfaction_mode`: `"global" | "area" | "knowledge" | "combination"`
- `satisfaction_baseline_alpha` (in `[0,1]`): EMA step size for satisfaction baseline

## Why This Design (Thesis Rationale)

The thesis scope excludes strategic voting and complex learning models. This altruism mechanism is:

- explicit (equation-driven)
- simple (few parameters)
- interpretable (a single “reality-vs-self” axis)

It provides a controlled way to create adaptive agents whose voting behavior can change over time,
without turning the thesis into a reinforcement learning project.

## What To Look At When Debugging

If altruism dynamics look surprising, inspect per-agent traces of:

- `altruism_factor`
- `dissatisfaction_value`, `dissatisfaction_baseline`, `dissatisfaction_signal`
- whether the agent actually participated that step (`participating`)

In schema v2 outputs:

- `agents.parquet` includes `altruism_factor`, `dissatisfaction_value`, `dissatisfaction_baseline`, `dissatisfaction_signal`
- `steps.parquet` includes `mean_altruism` and `mean_dissatisfaction`

## Test Coverage (What Is Locked By Pytests)

Individual contracts:

- `tests/test_altruism_static_contract.py` (static mode initialization + invariance + logging)
- `tests/test_altruism_init_contract.py` (learning-on initialization + validation)
- `tests/test_altruism_alpha_contract.py` (exact update, ratio metamorphic, logging)
- `tests/test_altruism_clip_bounds_contract.py` (oracle clipping + metamorphic tightening + validation)
- `tests/test_altruism_learning_toggle_contract.py` (toggle gates update, participant-only)
- `tests/test_satisfaction_mode_contract.py`, `tests/test_satisfaction_mode_validation_contract.py`,
  and `tests/test_satisfaction_baseline_alpha_contract.py`
  (dissatisfaction inputs to altruism)

Interaction tests (multi-knob, end-to-end through the step pipeline):

- `tests/test_altruism_learning_interactions.py`:
  - Baseline persistence: shows how `satisfaction_baseline_alpha` changes whether altruism keeps updating
    when dissatisfaction stays at its new level (1 update vs 2 updates in a controlled 3-step sequence).
  - Mode interaction: constructs a case where **global dissatisfaction is constant** but the **local area flips**;
    `satisfaction_mode="global"` yields zero signal (no update), while `"area"` yields a nonzero signal (update).
  - Clip interaction: forces a large positive signal and asserts the resulting altruism is clipped exactly at `altruism_clip_max`.
