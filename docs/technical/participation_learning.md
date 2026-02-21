# Participation Learning (Technical Contract)

This document is the single reference for **how participation learning works in code**:
what is updated, when it is updated, and why the model is designed this way.

## Where It Happens (Runtime Path)

Per model step (`t = 1,2,...`) the relevant call chain is:

1. `ParticipationModel.step()` calls `scheduler.step()`
2. `CustomScheduler.step()` iterates areas and calls `Area.step()`
3. `Area.step()`:
   - updates each agent’s knowledge + satisfaction (pre-election)
   - calls `Area.conduct_election()` which:
     - collects participation decisions + ballots (`_tally_votes`)
     - aggregates ballots using the voting rule
     - distributes rewards/costs (`_distribute_rewards`, then `VoteAgent.reward_agent`)
     - applies participation learning updates (this document)

Code references:

- Participation update rule: `src/agents/vote_agent.py::VoteAgent.apply_participation_update`
- Baseline + signal logic: `src/agents/area.py::Area.conduct_election`
- Participation decision logic: `src/agents/strategies.py::DefaultParticipationStrategy.decide_participation`

## State Variables (Per Agent)

Participation learning uses the following per-agent state:

- `q_participation` (float): internal propensity value (unbounded, but typically clipped)
- `participation_baseline` (float): EMA baseline of experienced outcomes (`delta_rel`), initialized as `NaN`
- `participation_signal` (float): **learning signal used for the update** (equals realized `delta_rel`)
- `participating` (bool): last action in the current election step (set during `_tally_votes`)

And the following per-election outcome signals:

- `fee` (float): participation cost for participants, `fee = election_cost_rate * assets_pre`
- `reward_personal` (float): outcome-dependent reward/penalty amount, computed in asset units
- `raw_delta_abs` (float): pre-clamp asset delta, `reward_personal - fee`
- `delta_abs` (float): realized absolute asset change after floor-clamp at zero assets
- `delta_rel` (float): realized relative change, `delta_abs / assets_pre` (if `assets_pre > 0`, else `0.0`)

`delta_abs`/`delta_rel` are computed and stored in `VoteAgent.reward_agent()` from the realized post-clamp asset change.

## Participation Decision (Policy)

The default decision policy is probabilistic:

1. Convert `q_participation` to a probability via a logistic function:

    ```text
    p = sigmoid(participation_beta * q_participation)
    sigmoid(x) = 1 / (1 + exp(-x))
    ```

2. Add an exogenous bias in **probability space** (not in logit space) and clip:

    ```text
    p' = clip(p + bias_toward_participation, 0, 1)
    ```

3. Sample participation with the model RNG:

    ```text
    participate ~ Bernoulli(p')
    ```

Design note: additive bias is a simple “civic duty / default norm” knob that is intentionally
separable from learning. It can be restricted in the UI (e.g. `[0, 0.5]`) even if the model
allows a broader conceptual range (e.g. `[-1, 1]`) for controlled baselines.

## Learning Signal: Realized Level Outcome

After the election is executed and the agent’s per-election `delta_rel` is computed,
the learning signal is:

```text
signal <- delta_rel
```

`participation_baseline_alpha` is an EMA step size:

- `0.0` means baseline never changes
- `1.0` means baseline becomes the last observed `delta_rel` immediately

Baseline maintenance is still tracked for diagnostics/logging:

```text
if baseline is NaN:
    baseline <- delta_rel
else:
    baseline <- (1 - participation_baseline_alpha) * baseline
               + participation_baseline_alpha * delta_rel
```

`participation_baseline` no longer changes learning behavior; it is an explanatory trace.

## Participation Update Rule (Reinforce Last Action)

The participation learning update is *action-reinforcement*, not counterfactual learning.
Agents reinforce whatever they just did, based on whether the experienced outcome was better/worse
than their recent baseline.

Define:

```text
sign = +1 if participating else -1
```

Then:

```text
q_participation <- q_participation + participation_alpha * sign * participation_signal
q_participation <- clip(q_participation, -participation_q_max, +participation_q_max)  (if q_max > 0)
```

Consequences (important for interpretation):

- If an agent participated and the signal is positive, `q` increases ⇒ the agent is more likely to participate again.
- If an agent abstained and the signal is positive, `q` decreases ⇒ the agent is less likely to participate again.
- This deliberately supports free-riding dynamics: abstention can be reinforced by good collective outcomes.

## Eligibility and “No Participation” Steps

Eligibility rule:

- Agents with `assets <= 0` are marked ineligible for the election and **do not** update learning that step.

All-abstain step:

- If no one participates in an area election, the model currently:
  - records turnout = 0
  - does **not** distribute rewards/costs
  - does **not** perform participation learning updates (because the learning loop is after reward distribution)

This is a semantics decision. If you later want “learning even when no one participates”, it should be an explicit,
audited change (not an accidental side effect).

## Knobs (What They Mean)

Participation learning knobs (ModelConfig):

- `participation_alpha` (>= 0): learning rate in q-space (step size of updates)
- `participation_beta` (>= 0): sensitivity of probability to q (steepness of sigmoid)
- `participation_init_q` (finite): initial q for all agents
- `participation_q_max` (>= 0): symmetric clipping bound for q; `0` disables clipping
- `bias_toward_participation` (in `[-1,1]`): additive probability bias after sigmoid, then clipped to `[0,1]`
- `participation_baseline_alpha` (in `[0,1]`): EMA step size for the logged participation baseline trace

Scale note (relative signals):

- Since learning uses `delta_rel` (small, typically ~1e-3 to 1e-2), q updates are small.
- Practical ranges are correspondingly smaller: `participation_q_max` in ~`[1, 3]` and `participation_init_q` in ~`[-3, 3]`.
- `participation_beta` is the main amplifier if learning feels too slow under small signals.

Practical intuition:

- `alpha` mostly sets “how fast behavior changes”
- `beta` mostly sets “how decisive q becomes” (low beta ≈ always ~0.5, high beta ≈ near-deterministic)
- baseline alpha only affects the diagnostic baseline trace, not participation updates

## Why This Design (Thesis Rationale)

This model is intentionally a “public good participation” setting:

- Outcomes (common component) affect *everyone*, including abstainers.
- Participation is costly (fee), but outcome benefits are mostly non-excludable.

The learning rule is deliberately:

- **simple** (few parameters; auditable)
- **behaviorally plausible** (reinforcement of experienced outcomes)
- **not strategically optimal** (no counterfactual reasoning; no strategic voting)

This aligns with the thesis scope: time-dependent dynamics under different voting rules with explicit adaptation,
without modeling sophisticated strategic reasoning.

## What To Look At When Debugging

For any surprising turnout dynamics, inspect per-agent traces of:

- `participating`
- `fee`
- `delta_abs`, `delta_rel`
- `participation_baseline`, `participation_signal`
- `q_participation`, `p_participation`

In schema v2 outputs:

- `agents.parquet` contains `participation_baseline` and `participation_signal` (and can be extended with q/p if desired).

## Test Coverage (What Is Locked By Pytests)

The following tests lock the contract:

- Unit/oracle + metamorphic tests for individual knobs:
  - `tests/test_participation_alpha_contract.py`
  - `tests/test_participation_beta_contract.py`
  - `tests/test_participation_init_q_contract.py`
  - `tests/test_participation_q_max_contract.py`
  - `tests/test_bias_toward_participation_contract.py`
  - `tests/test_participation_baseline_alpha_contract.py`
  - `tests/test_no_participation_debug_snapshot.py` (no-participation step keeps participation debug semantics consistent)
- Interaction tests exercising multi-knob behavior under controlled RNG:
  - `tests/test_participation_learning_interactions.py`:
    - Level-signal contract: `participation_signal == election_delta_rel` and q-updates are baseline-alpha invariant.
    - Clipping interaction: demonstrates repeated updates drive `q_participation` until it hits `±participation_q_max`.
    - Beta sensitivity: holds the same q and the same RNG draw fixed, and shows that higher
      `participation_beta` can flip the next participation decision (more sensitivity to q).
