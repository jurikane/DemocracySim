# Grid State Concept

This is to explain the conceptual role of the color grid as implemented in the thesis model.

## 1. What the Grid Represents

The grid is the realized environment state of the simulation.
It is not only visualization; it is a causal state variable that stores the cumulative effect of prior collective decisions.

Interpretation:

- the grid is social memory of past elections
- it encodes what direction the system has actually moved to
- it provides the state against which current agent satisfaction is evaluated

## 2. How the Grid Changes (Implemented Mechanics)

Per area, elections produce an elected ordering (`voted_ordering`).
Color mutation is then applied from that ordering with mutation rate `mu` and color-selection probabilities derived from election impact settings.

Important timing:

- election and reward happen at step `t`
- mutation from that election is applied at the beginning of step `t+1`

So the grid has lagged dynamics: decisions shape future environment, not the same instant state.

## 3. Why This Matters Conceptually

This lag creates a feedback structure that is central to the thesis:

- agents vote under current state constraints
- decision outcomes update resource signals and learning
- only later does the environment itself move

## 4. Grid and Satisfaction

Agent dissatisfaction is computed by comparing each agent's preferred distribution to a target distribution (typically the current area grid distribution in thesis baseline).

Therefore:

- grid movement changes satisfaction distribution across agents
- satisfaction then changes altruistic-vote probability under satisfaction mode
- this links realized state history to future ballot composition

## 5. Scope and Limits

The grid is a stylized environment model.
It does not claim to represent a specific real-world socio-economic state variable.
Its role is to provide a coherent, path-dependent state for controlled comparative dynamics.

## 6. Relation to Thesis Question

Because voting rules affect election outcomes and outcomes affect grid evolution, voting rules can indirectly shape:

- participation trajectories
- resource inequality trajectories
- experiential inequality trajectories

This is one of the key channels through which rule differences become measurable over time.
