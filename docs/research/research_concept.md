DemocracySim is set in a grid-based environment where agents interact with their surroundings and participate in group decision-making through elections. The system explores various scenarios and voting rules to understand key dynamics and challenges in democratic participation.

## Key Features

### Simulated Environment:
- The grid is designed without boundaries, and each unit (field) within it adopts one of **x** colors. Fields change color based on election results, with a mutation rate affected by prior outcomes.
- Groups of fields form **territories**, which serve as the basis for elections and influence grid evolution.

### Agents:
- Agents are equipped with a basic artificial intelligence system and operate under a **"top-down" model**, learning decision-making strategies via training.
- Each agent has a **limited budget** and must decide whether to participate in elections.
- Agents have individual **preferences** over colors (called *personalities*) and are divided into **y** randomly distributed personality types.  
  *(The distribution of types forms majority-minority situations.)*

### Elections and Rewards (Two Dilemmas):
1. **Elections:**
    - Elections concern the frequency distribution of field colors in a given territory, representing an "objective truth" aimed at emulating wise group decisions.
    - For an intuitive understanding, the election addresses the question:  
      *"What is — or should be — the current color distribution within your territory?"*

2. **Rewards:**
    - Rewards are distributed to all agents in the territory, regardless of participation (*participation dilemma*).  
      These rewards consist of:
        - **Base reward:** Distributed equally based on how well agents guess the true color distribution.
        - **Personal reward:** Allocated based on the alignment between election results and agent preferences, introducing a second dilemma:
            - *Should agents vote selfishly (favoring their preferences) or vote with a focus on the group's accuracy (collective good)?*

