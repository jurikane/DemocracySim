# Simulation Environment

The environment in DemocracySim is structured as a grid, where elections and agent interactions occur.

---

### Features
1. **Dynamic Environment**:
    - The grid is composed of cells, each representing a state with a specific color.
    - Elections in areas of the grid drive state transitions.

2. **Mutations and Elections**:
    - Mutations are applied to the grid, introducing randomness.
    - Voting outcomes reflect agent personalities and decision-making.

---

#### Area Workflow

```mermaid
sequenceDiagram
    participant Area
    participant Election
    participant Agent
    Area->>Election: Area holds an election
    Area->>Agent: Updates each agents knowledge
    Agent->>Area: Has knowledge
    Agent->>Election: Participate in election
    Election->>Agent: Receive reward
    Agent->>Agent: Update assets and strategies
```

