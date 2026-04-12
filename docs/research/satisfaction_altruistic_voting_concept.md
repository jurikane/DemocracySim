# Satisfaction to Altruistic Voting Concept

This is to explain how dissatisfaction, satisfaction, and altruistic ballot probability are linked in the current model.

## 1. Behavioral Design Choice

Agents do not learn ballot strategies directly.
The learned/updated behavioral axis for the thesis is participation.

Ballot mode is still heterogeneous per vote, but controlled through a state variable (`altruism_factor`) rather than strategic optimization.

## 2. Dissatisfaction Signal

Before election each step, each agent computes `dissatisfaction_value` as distance between:

- agent preferred distribution (`personal_opt_dist`)
- target distribution from `satisfaction_mode`

Supported target modes:

- `area` (default and used in thesis runs)
- `global`
- `knowledge`
- `combination`

Baseline trajectory:

- dissatisfaction baseline is EMA-updated
- `dissatisfaction_signal = dissatisfaction_value - dissatisfaction_baseline`

In thesis baseline, dissatisfaction level itself is used to map altruism in satisfaction mode.
It is not learned nor strategic.

## 3. Satisfaction-Mode Mapping to Altruism

If `altruism_mode=satisfaction`, the mapping is:

- `s = 1 - dissatisfaction_value`
- `target = sigmoid(k * (s - theta))`
- with response smoothing parameter `gamma`

Then `altruism_factor` is clipped into configured bounds.

Interpretation:

- higher satisfaction tends to raise altruism probability
- lower satisfaction tends to lower altruism probability

## 4. Vote-Time Mode Selection

During vote casting, a Bernoulli draw is made using `altruism_factor`:

- altruistic mode with probability `altruism_factor`
- self-regarding mode otherwise

Ballot construction:

- altruistic mode: use estimated distribution from limited knowledge (`known_cells`), convert to ordering, then score options against that ordering
- self-regarding mode: use static precomputed oppose-scores from preference-group ordering

This keeps vote-mode heterogeneity while avoiding strategic game-policy complexity.

## 5. Why This Concept Exists

The thesis needs a mechanism linking environment fit to collective-choice quality capacity.
This mapping provides that channel:

- state fit (satisfaction) influences ballot orientation
- ballot orientation influences quality-gate success probability
- quality sign influences reward regime and subsequent participation learning
- fixed shift of satisfied voters to altruistic vote can limit lock-in states

## 6. Limits

This is not a model of explicit moral reasoning.
"Altruistic" here means voting according to estimated common-quality signal rather than self-regarding preference.
