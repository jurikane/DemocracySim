# Grid State Concept

The color grid is more than a visualization layer. It is the realized
environment state of the model.

## 1. What the Grid Represents

The grid stores the cumulative effect of prior collective decisions. In that
sense, it is social memory of past elections, it records what direction the
system has actually moved in, and it provides the state against which current
agent satisfaction is evaluated.

## 2. How the Grid Changes (Implemented Mechanics)

Per area, elections produce an elected ordering (`voted_ordering`).
Color mutation is then applied from that ordering with mutation rate `mu` and color-selection probabilities derived from election impact settings.

Timing matters here: election and reward happen at step `t`, while mutation
from that election is applied at the beginning of step `t+1`.

So the grid has lagged dynamics: decisions shape future environment, not the same instant state.

## 3. Why This Matters Conceptually

This lag creates a feedback structure that is central to the simulation:
agents vote under current state constraints, decision outcomes update resource
signals and learning, and only later does the environment itself move.

## 4. Grid and Satisfaction

Agent dissatisfaction is computed by comparing each agent's preferred distribution to a target distribution (typically the current area grid distribution).

Grid movement changes the satisfaction distribution across agents, satisfaction
then changes altruistic-vote probability under satisfaction mode, and realized
state history thereby feeds into future ballot composition.

## 5. Scope and Limits

The grid is a stylized environment model.
It does not claim to represent a specific real-world socio-economic state variable.
Its role is to provide a coherent, path-dependent state for controlled comparative dynamics.

## 6. Relation to Thesis Question

Because voting rules affect election outcomes and outcomes affect grid evolution, voting rules can indirectly shape:

- participation trajectories
- resource inequality trajectories
- experiential inequality trajectories

This is one of the main channels through which rule differences become
measurable over time.
