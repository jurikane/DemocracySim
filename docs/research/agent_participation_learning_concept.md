Participation Learning Under Public-Good Dynamics
=================================================

A central challenge in modeling electoral participation is that voting is a
public good. Individual participation is costly, while the benefits of good
collective decisions are largely non-excludable. As a result, rational
free-riding may well be locally attractive, even when widespread participation
is socially beneficial.

In real democratic systems, non-participants still benefit from outcomes
shaped by others’ participation. Consequently, learning signals related to
participation are inherently noisy and correlation-based, rather than cleanly
action-conditional.

This project deliberately embraces this structure.


Intentional Design Choice: Rewarding Abstainers
-----------------------------------------------

In the model, all agents — participants and non-participants alike — receive
outcome-based rewards derived from the quality of the elected outcome relative
to societal reality and personal preferences. Participation is penalized only
through an effort-like cost proportional to agent assets.

This means that:
- Abstainers may receive positive rewards when others participate.
- Participants may receive negative rewards when collective outcomes are poor.
- The learning signal is not a counterfactual “what would have happened if I
  had acted differently,” but an experienced outcome.

This mirrors real-world political learning:
- Individuals may reinforce abstention when benefitting from it.
- They may abandon participation after it turned out unsuccessful for them.
- Participation decisions are influenced by perceived collective performance,
  social norms (bias-toward-participation), and past outcomes.


Learning as Reinforcement of Past Behavior
------------------------------------------

The participation learning mechanism is intentionally simple and
psychologically plausible:

- Agents track an internal participation propensity.
- After each election, the realized outcome modifies this propensity.
- Positive outcomes reinforce the agent’s last decision (participate or
  abstain).
- Negative outcomes weaken it.

This mechanism is not intended to produce optimal individual behavior.
Instead, it captures a form of behavioral reinforcement learning under social
uncertainty, where agents adapt based on perceived success rather than causal
attribution.


Herding, Free-Riding, and Instability Are Expected Outcomes
-----------------------------------------------------------

This design naturally allows for:
- Free-rider dynamics.
- Herding effects.
- Participation cycles and instabilities.
- Sensitivity to population composition, voting rules, and incentive
  parameters.

These phenomena are not failures of the model. They are precisely what the
model is designed to explore. The core research question is not whether agents
learn “correctly,” but whether and under what conditions participation can
emerge or collapse in the presence of public-good incentives.


Controlling for Social Norms
----------------------------

To reflect institutional or cultural factors (e.g., civic duty or default
voting norms), the model may include an exogenous participation bias that nudges
agents toward participation independently of learning.

This bias:
- Does not eliminate free-riding.
- Does not replace learning.
- Serves as a tunable baseline for comparative experiments.


Alternative Learning Regimes as Extensions
------------------------------------------

More causally sophisticated learning rules (e.g., participant-only updates or
advantage-based learning) may be implemented as alternative regimes for
comparison. However, the baseline model intentionally prioritizes behavioral
plausibility and social realism over individual optimality.