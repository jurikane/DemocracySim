# Dynamics and Voting-Rule Interaction Concept

The thesis rests on a closed feedback loop: rule choice changes election
outcomes, outcomes change incentives and learning, and those changes reshape
later elections.

## 1. Closed-Loop Dynamic

The implemented loop links rules to participation through multiple channels:

1. Rule aggregates mixed ballots into elected ordering.
2. Elected ordering determines quality-gate sign and reward structure.
3. Reward and fee components update participation learning signals.
4. Participation composition in later steps changes ballot pools.
5. Election outcomes shape later grid state (lagged mutation), which affects dissatisfaction and altruistic-vote probability.

This creates endogenous temporal dynamics where rule differences can accumulate.

## 2. Power-Struggle vs Quality-Alignment Tension

The model intentionally allows two directional pressures:

- self-regarding ballots that push toward group preference power
- altruistic ballots that use distributed knowledge to track quality pressure

Rule aggregation behavior determines how these pressures are combined each step.

## 3. Satisfying-Region Attractor Intuition (Design-Level)

If the system reaches a region where many agents are relatively satisfied,
altruistic-vote probability tends to increase, which can improve quality-gate
alignment and reduce destabilizing punishment regimes.

At the same time, self-regarding pressure can prevent convergence or create lock-in/polarized movement.

In the thesis, this attractor logic is treated as design intuition and
exploratory expectation rather than as a primary confirmatory hypothesis.
