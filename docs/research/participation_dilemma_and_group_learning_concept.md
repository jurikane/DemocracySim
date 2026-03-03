# Participation Dilemma and Group-Relative Learning Concept

This is to explain why participation learning is group-relative in the thesis baseline.

This document complements:

- `docs/research/participation_learning_group_relative_party_mode.md`

## 1. Participation Dilemma Built Into the Model

The model intentionally combines:

- participation fee paid only by participants
- collective reward/punishment mode based on decision quality gate
- reward magnitude tied to preference alignment
- both fees and rewards are relative to agents assets

Consequence:

- participation cost is individually concentrated
- quality sign is socially shared
- within personality groups, reward components are identical, while fee remains an individual participation signal

This creates a structured free-rider tension.

## 2. Why Fees and Rewards Are Not Absolute but Relative to Assets

This was a conscious choice to account for the relative nature of effort and reward.
The model does not attempt to represent direct real-world wealth-power effects.
Instead, it models motivational resources and reward pressure in relative terms.
Absolute differences still remain analytically visible and comparable.

## 3. Why Pure Individual Reward Learning Is Weak Here

If participation learning uses only immediate personal reward-like signals, the effective differentiating information can collapse toward fee effects, reducing meaningful group-competition structure.

For this thesis trajectory, learning is therefore framed around relative group performance under participation cost.

## 4. Group-Relative Party Mode (Baseline)

Mode: `participation_signal_mode=group_relative_delta_rel_party`

Definitions for each step:

- `mu_g`: mean `election_delta_rel` among eligible agents in group `g`
- `mu_bar`: mean of `mu_g` across groups with eligible members
- `r_g = w_g * (mu_g - mu_bar)`
- `w_g = n_g / (n_g + participation_signal_group_shrink_k)`

Per-agent signal:

- `signal_i = clip(r_g + fee_component_i, +/- participation_signal_clip)`
- participant fee term: `fee_component_i = - participation_signal_fee_weight * fee_rel_i`
- abstainer fee term: `fee_component_i = 0`

## 5. Update Direction for Participants and Abstainers

In this baseline mode, for every eligible agent `i` (participant or abstainer):

- `q_i <- q_i + participation_alpha * signal_i`

So both participants and abstainers receive the same group-relative directional component `r_g`.
Only participants receive the additional negative fee component.

Implication:

- group-level direction can move both subpopulations similarly
- participant updates are damped (or locally reversed) when fee salience dominates

## 6. Why This Fits the Thesis Question

The thesis asks how voting rules affect participation dynamics.
This learning design increases sensitivity of participation updates to collective comparative outcomes, making rule effects on participation more observable and interpretable.

## 7. Scope Boundary

This is a stylized party-relative learning rule.
It is not a claim that real voters update behavior exactly this way.
