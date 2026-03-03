DemocracySim is a grid-based multi-agent simulation where agents participate in repeated area-level elections and adapt behavior over time.
The project studies how voting rules shape participation and inequality dynamics.

## Concept Summary

### Environment

- The world is a toroidal color grid.
- Areas (territories) conduct elections on local color distributions.
- Collective decisions feed into mutation dynamics that alter later election states.

### Agents

- Agents have heterogeneous preference structures (personality groups + per-agent preference distributions).
- Agents hold a resource/capacity state (`assets`) and decide whether to participate in each election.
- Agents have limited information (`known_cells`) and estimate local reality with uncertainty.
- Adaptation uses fixed explicit update rules (participation learning and optional altruism learning).

### Elections and Incentives

- Elections aggregate agent score vectors into a collective ordering via a chosen voting rule.
- Economic update per election combines:
  - participation fee (`election_cost_rate`)
  - common reward/penalty (reality distance based)
  - personal reward/penalty (preference distance based)
- Signs are controlled by break-even distances; magnitudes scale with current assets.

### Core Tensions

- Participation dilemma: participate and pay immediate cost vs abstain and rely on others.
- Alignment dilemma: prioritize personal preference fit vs collective reality tracking.
- Dynamic feedback: election outcomes affect both immediate payoffs and future environment states.
