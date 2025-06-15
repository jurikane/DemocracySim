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
