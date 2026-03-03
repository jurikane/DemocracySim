# Dynamics and Voting-Rule Interaction Concept

This is to document the system-level dynamic intuition behind the thesis without overstating confirmatory claims.

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

Design intuition:

- if the system reaches a region where many agents are relatively satisfied,
  altruistic-vote probability tends to increase,
  which can improve quality-gate alignment,
  which in turn can reduce destabilizing punishment regimes.

At the same time, self-regarding pressure can prevent convergence or create lock-in/polarized movement.

Thesis framing decision:

- this attractor logic is treated as theoretical design intuition and exploratory expectation
- it is not a primary confirmatory hypothesis by default

## 4. Why Voting Rules Matter Here

Voting rules can differ in how they:

- translate heterogeneous ballots into winners
- mediate majority/minority influence
- preserve or suppress minority directional information

Therefore, rule choice can change both:

- short-run election quality outcomes
- long-run participation and inequality trajectories
