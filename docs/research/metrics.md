## Simulation Metrics / Indicators

For final thesis analysis, the frozen operational definitions are documented in
`docs/research/thesis_measurement_spec.md`.

### **Participation Rate** *(Aggregate Behavioral Variable)*

- Measures the percentage of agents actively participating in elections at a given time.
- Helps evaluate the *participation dilemma* by analyzing participation across the group and comparing rates for majority vs. minority groups.

### **Altruism Factor** *(Mechanism Variable)*

- Quantifies the extent to which agents prioritize the **collective good** (e.g., the group's accuracy in guessing) over **individual preferences**, including cases of non-cooperation with a majority they belong to when it conflicts with the (expected) collective good.
- Additionally, tracking the average altruism factor of personality groups can provide insights, though this may be misleading if agents/groups do not participate.

### **Gini Index** *(Inequality Metric)*

- Measures inequality in agent resource/capacity state (assets).
- In the simulation outputs (`steps.parquet`, `area_steps.parquet`) this is stored as
  a percentage-like value in **0–100** (`100 * gini`).
- Interpretation is unchanged: **0** = perfect equality, **100** = maximum inequality.
- For thesis analysis, inequality is two-dimensional:
  - `gini_assets` (resource dimension)
  - `gini_dissatisfaction` (experiential dimension from `satisfaction_value`)

### **Collective Accuracy / Reality Distance**

- Measures how accurately the group, as a collective, estimates the actual color distribution.
- This directly influences rewards and serves as a metric for evaluating group performance against a ground truth.

### **Diversity of Shared Opinions** *(Optional Descriptive)*

- Evaluates the variation in agents' expressed preferences.
- Tracks whether participating agents provide diverse input or converge to similar opinion patterns.

### Out of Scope for This Thesis Baseline

- Normative “distance-to-optimum” criteria (utilitarian/egalitarian/Rawlsian) are not used as baseline evaluation targets.
