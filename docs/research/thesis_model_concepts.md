# Thesis Model Concepts

This document explains the conceptual logic behind the simulation as currently implemented.
It is intentionally narrative and causal (not only metric/schema focused).

Related technical contracts:

- `docs/research/thesis_contract.md`
- `docs/research/thesis_measurement_spec.md`
- `docs/research/participation_learning_group_relative_party_mode.md`

Detailed concept notes:

- `docs/research/grid_state_concept.md`
- `docs/research/puzzle_quality_gate_concept.md`
- `docs/research/satisfaction_altruistic_voting_concept.md`
- `docs/research/participation_dilemma_and_group_learning_concept.md`
- `docs/research/dynamics_rule_interaction_concept.md`

## 1. Core Purpose

The model studies how voting-rule differences shape participation and inequality dynamics under a fixed adaptive environment.

The central behavioral target is participation: agents only learn whether to participate in elections.
They do not learn strategic ballot manipulation.

## 2. Core Entities and Representations

- Agents belong to personality groups defined by preference orderings over colors.
- Each agent also has a per-agent preferred color distribution (`personal_opt_dist`) consistent with that ordering.
- Elections aggregate participant ballots into a winning color ordering.
- Assets represent a generic resource/capacity state (not literal money).

Key implication:

- Resource inequality can emerge endogenously even if initialization is equal.

## 3. Two-Mechanism Environment: Grid and Puzzle

The current model separates two environmental roles that were previously conflated:

1. Grid state (realized world state)

    - The grid is the realized color distribution shaped by past election outcomes through mutation.
    - It is path-dependent memory of collective decisions.
    - Agent satisfaction/dissatisfaction is computed against this realized state (depending on satisfaction mode).

2. Puzzle Quality Gate (quality reference process)

    - In `quality_target_mode=puzzle`, each area maintains a Puzzle Quality Gate distribution on the simplex.
    - The puzzle process evolves via a local Dirichlet random walk with occasional redraw shocks.
    - This process is color-symmetric (no fixed directional bias by color label).
    - Decision quality is evaluated against Puzzle Quality Gate alignment (not directly against current grid ordering).

Conceptual role of the split:

- Grid = what society has currently become.
- Puzzle Quality Gate = external directional reality, conditioning or window of opportunity / quality gate for whether decisions are rewarded.

## 4. Election-to-Learning Causal Loop (Per Step)

For each area and step, the implemented order is:

1. Update puzzle state (if puzzle mode), otherwise clear puzzle state.
2. Update each agent's known information (`known_cells`).
   - In puzzle mode, knowledge samples are drawn from puzzle distribution.
   - In reality mode, knowledge samples come from area cells.
3. Compute dissatisfaction (`dissatisfaction_value`) and baseline/signal updates.
4. If `altruism_mode=satisfaction`, map dissatisfaction to `altruism_factor` via sigmoid response.
5. Agents decide participation probabilistically from learned `q_participation`.
6. Participants pay election fee and cast ballots.
7. Ballot mode is sampled per vote:
   - altruistic ballot with probability `altruism_factor`
   - otherwise self-regarding ballot
8. Voting rule selects winning ordering.
9. Quality gate computes decision quality:
   - puzzle mode: `puzzle_distance`
   - reality mode: `dist_to_reality`
10. Reward/punishment sign is binary by quality threshold; magnitude scales with group alignment.
11. Participation learning updates `q_participation` from group-relative signal and fee salience.
12. Mutation from election outcome is applied at the start of the next scheduler step.

Important timing consequence:

- Elections/rewards happen on the current election-time state.
- Grid mutation is lagged to the next step.

## 5. Satisfaction-Driven Altruistic Voting Concept

In satisfaction mode, altruistic voting is not an explicit strategic choice variable.
Instead:

- Dissatisfaction is computed as distance between agent preference distribution and target distribution (baseline mode usually `area`).
- Satisfaction proxy is `1 - dissatisfaction`.
- `altruism_factor` is updated by sigmoid mapping with threshold/slope and optional smoothing. This enables it to free simulations from too many majority deadlocks.
- During vote casting, `altruism_factor` is used as the probability of voting altruistically.

Interpretation:

- Higher satisfaction tends to increase altruistic-vote probability.
- Lower satisfaction tends to increase self-regarding (power-struggle) voting.

## 6. Participation Learning Concept (Why Group-Relative Party Signal)

The model intentionally includes participation cost and non-excludable outcome effects. Rewards and fees are intentionally relative to assets.
This creates a free-rider structure:

- Participation is costly (fee paid only by participants).
- Reward/punishment mode is collective (quality gate), not participation-conditional.
- Within personality groups, reward components are strongly shared; fee is the key individual difference.

Naive individual reward-learning can collapse into synchronized or weakly informative participation dynamics.

Therefore, the implemented participation-learning mode (`group_relative_delta_rel_party`) uses:

- group relative performance vs other groups
- group-size shrinkage
- explicit participant fee salience
- direct q-space updates for all eligible agents:
  - for each eligible agent `i`, `q_i <- q_i + participation_alpha * signal_i`
  - participants and abstainers both receive the group-relative component
  - participants additionally receive a fee penalty component in `signal_i`

Conceptual effect:

- learning is framed as "how my group performs relative to others" instead of only "my personal absolute reward".

## 7. Decision-Quality and Social-Tension Interpretation

The quality gate formalizes a "decision viability" pressure:

- If elected ordering aligns sufficiently with quality reference, rewards are positive mode.
- If not, negative mode applies.

At the same time, self-regarding voting can pull outcomes away from this reference.
This creates tension between:

- puzzle-solving / collective alignment behavior
- power-struggle / preference-dominance behavior

Voting rules are expected to affect this balance by how they aggregate mixed ballots under heterogeneous groups.

Attractor Design Intuition (Hypothetical):

The puzzle’s "rewardable direction" drifts randomly overall.
If the grid state yields high satisfaction, agents vote more puzzle-aligned (via altruism), so outcomes tend to track that random drift rather than being pulled toward fixed group extremes.

This feedback could, in principle, create a weak attractor:

- broadly satisfying grid states become self-stabilizing (more altruism => better puzzle alignment => fewer negative shocks).
- dissatisfaction increases power-struggle voting, which can disrupt stability and induce lock-ins/oscillations.
- voting rules may change how strongly such disruptions prevent convergence toward broadly satisfying regions.

## 8. Why This Supports the Thesis Question

The model design ties voting-rule differences to participation dynamics through a closed loop:

- Rule affects election outcomes.
- Outcomes affect quality sign and reward distribution.
- Reward and fee signals affect participation-learning updates.
- Participation composition affects future elections.

This makes participation and inequality trajectories endogenous to rule choice under fixed non-rule mechanics.

## 9. Explicit Non-Goals

The model does not claim to represent:

- strategic game-theoretic voting equilibria
- empirically calibrated real-world democracy
- normative proof of optimal democratic design

It is a controlled simulation framework for comparative dynamics.
