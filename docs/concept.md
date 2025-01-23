# Concept

**DemocracySim** is a multi-agent simulation framework designed to explore the effects of different voting rules on democratic participation and welfare. Developed as part of a master's thesis at Leipzig University, the project investigates how collective decision-making processes shape individual participation, resource distribution, and long-term system dynamics. With a focus on agent-based modeling, the simulation ties together elements of participatory dynamics, resource allocation, and group decision effects in a controlled, evolving system.

---

## Project Summary

DemocracySim is set in a grid-based environment where agents interact with their surroundings and participate in group decision-making through elections. The system explores various scenarios and voting rules to understand key dynamics and challenges in democratic participation.

### Key Features

#### Simulated Environment:
- The grid is designed without boundaries, and each unit (field) within it adopts one of **x** colors. Fields change color based on election results, with a mutation rate affected by prior outcomes.
- Groups of fields form **territories**, which serve as the basis for elections and influence grid evolution.

#### Agents:
- Agents are equipped with a basic artificial intelligence system and operate under a **"top-down" model**, learning decision-making strategies via training.
- Each agent has a **limited budget** and must decide whether to participate in elections.
- Agents have individual **preferences** over colors (called *personalities*) and are divided into **y** randomly distributed personality types.  
  *(The distribution of types forms majority-minority situations.)*

#### Elections and Rewards (Two Dilemmas):
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

---

## Simulation Metrics / Indicators

### **Participation Rate** *(Aggregate Behavioral Variable)*
- Measures the percentage of agents actively participating in elections at a given time.
- Helps evaluate the *participation dilemma* by analyzing participation across the group and comparing rates for majority vs. minority groups.

### **Altruism Factor** *(Individual Behavioral Variable)*
- Quantifies the extent to which agents prioritize the **collective good** (e.g., the group's accuracy in guessing) over **individual preferences**, including cases of non-cooperation with a majority they belong to when it conflicts with the (expected) collective good.
- Additionally, tracking the average altruism factor of personality groups can provide insights, though this may be misleading if agents/groups do not participate.

### **Gini Index** *(Inequality Metric)*
- Measures the inequality in asset distribution among agents within the system.
- Ranges from **0** (perfect equality) to **1** (maximum inequality, where one agent holds all assets).
- Offers insights into how electoral decisions impact wealth/resource distribution over time.

### **Collective Accuracy**
- Measures how accurately the group, as a collective, estimates the actual color distribution.
- This directly influences rewards and serves as a metric for evaluating group performance against a ground truth.

### **Diversity of Shared Opinions**
- Evaluates the variation in agents' expressed preferences.
- To track whether participating agents provide diverse input or converge on overly similar opinions (e.g., due to majority influence).

### **Distance to Optimum**
In principle, the optimal decision can be determined based on a predefined goal, allowing the distance between this optimum and the group's actual decision to be measured.

**Possible predefined goals include:**

1. **Utilitarian**:
    - *Maximize the total sum of distributed rewards.*
    - Focus on the *total reward*, regardless of how it is distributed.

2. **Egalitarian**:
    - *Minimize the overall inequality in individual rewards.*
    - Focus on **fairness**, aiming for a more just distribution of rewards among members.

3. **Rawlsian**:
    - *Maximize the rewards for the poorest (personality-based) group.*
    - Inspired by **John Rawls' Difference Principle**, the focus is on improving the well-being of the least advantaged group while tolerating inequalities elsewhere.

---

## Research Questions

DemocracySim seeks to answer several critical questions:

- Do different voting procedures produce varying dynamics, and if so, how?
- How do minority and majority agent types behave in collective decision-making?
- What are the long-term effects of (non-)participation on the system?
- How does wealth distribution impact participation and welfare in the simulation?

---

## Broader Implications

This project offers a controlled testbed for understanding the complex interplay of individual and collective interest in democratic systems. DemocracySim has the potential to reveal valuable insights into real-world voting dynamics.
