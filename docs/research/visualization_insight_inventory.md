# Thesis-First Visualization Insight Inventory (DOE/HIL/Debug)

This document is the canonical planning/inventory layer for simulation visualizations.
It is intentionally **insight-first** (thesis intent and model concepts) and only then mapped to current plots.

Core inputs used for this inventory:

- `docs/research/thesis_contract.md`
- `docs/research/thesis_model_concepts.md`
- `docs/research/thesis_measurement_spec.md`
- `docs/research/execution_scope_freeze.md`
- `docs/research/metric_glossary.md`
- `docs/research/participation_learning_group_relative_party_mode.md`
- `docs/technical/participation_learning.md`
- `docs/technical/api/output_schema_v2.md`
- `src/analysis/summary_tooling.py`
- `src/analysis/doe_review_bundle.py`

---

## 1. Operating Modes (Planning Baseline)

Two modes are required and should remain explicitly separated:

- `debug_doe`:
  maximize causal observability and failure diagnosis during DOE tuning/HIL.
- `thesis_run`:
  minimize to confirmatory evidence and compact supporting diagnostics for final reporting.

Rule:

- If a plot does not answer a required `thesis_run` question, it must stay `debug_doe` only.

---

## 2. Insight Inventory (Ordered by Need-to-Have Insight)

The following list is the primary structure for all future plot decisions.
Plots are implementation choices; insights are the contract.

| Insight ID | Need-to-understand question | Why required | Required mode(s) | Minimum evidence | Current coverage | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `I-01` | Are participation and inequality trajectories meaningfully different over time? | Core thesis outcomes (`turnout`, `gini_assets`, `gini_dissatisfaction`) | `debug_doe`, `thesis_run` | global time series with stable scales | Strong | Keep |
| `I-02` | Is decision quality improving/degrading and under what diversity regime? | Quality trajectory and tension with diversity (`dist_to_reality`, entropy) | `debug_doe`, `thesis_run` | global + area quality traces + diversity | Strong | Keep |
| `I-03` | How close are outcomes to fixed reference optima families? | Benchmark lens (`dist_to_ref_*`) is frozen analysis contract | `debug_doe`, `thesis_run` | distance trajectories vs utilitarian/nash/egalitarian/rawlsian | Strong | Keep, compact in thesis mode |
| `I-04` | Is puzzle-vs-power tension present and does anti-monopoly behavior hold? | DOE quality gates and model concept of puzzle/power conflict | `debug_doe` (primary), `thesis_run` (optional appendix) | decomposition + gate page | Strong | Keep in debug; optional in thesis |
| `I-05` | Do group opportunity differences align/misalign with observed group behavior? | Majority/minority dynamics are a key descriptive lens | `debug_doe` | opportunity distance + behavior spread | Medium-strong | Keep, simplify legends |
| `I-06` | Are turnout/composition differences across groups genuine (not tiny-group noise)? | DOE/HIL decision quality depends on readable group diagnostics | `debug_doe` | group turnout/composition/switch/dropout with small-group handling | Strong (recently improved) | Keep with pruning |
| `I-07` | Is participation learning causality interpretable end-to-end? (`mu_g`, `mu_bar`, group component, fee component, final signal, resulting `Δq`) | Thesis model concept explicitly depends on this loop | `debug_doe` (must), `thesis_run` (summarized) | one causal decomposition view with exact components | **Weak / fragmented** (spread across multiple proxy pages) | **New/replace required** |
| `I-08` | Is altruism-learning behavior interpretable and separated by vote mode? | Needed to explain altruistic/self-regarding dynamics under satisfaction mapping | `debug_doe` | signal + update shift by group and vote mode | Medium | Keep, integrate with I-07 narrative |
| `I-09` | Do gates/stability constraints pass for substantive reasons, not artifacts? | DOE gate validity and freeze discipline | `debug_doe` | explicit gate traces + support counts + lock-in/chaos context | Medium | Keep, attach support context |
| `I-10` | Which knobs matter for which metrics, where in range optima are, and with what confidence | DOE tuning objective and HIL comparability | `debug_doe` | knob-metric correlation/effect pages + rank packet context | Medium | Keep, extend with uncertainty next |
| `I-11` | Can a reviewer map a PDF to exact run/design/knob settings and DOE ranges quickly? | HIL productivity + traceability requirement | `debug_doe` | packet overview + queue table + direct links | Strong (improved) | Keep |
| `I-12` | Are diagnostics numerically trustworthy (coverage, NaN policy, low-support flags)? | Prevent misleading interpretation in sparse steps/groups | `debug_doe`, `thesis_run` | explicit support/coverage cues | Medium | Expand modestly |

---

## 3. Plot Family Inventory (Current -> Target)

This maps current plot families to insight IDs and action decisions.

| Plot ID | Current plot / page title | Insight IDs | `debug_doe` | `thesis_run` | Action |
| --- | --- | --- | --- | --- | --- |
| `P-G-CORE-01` | `Global Core Metrics` | `I-01` | Keep | Keep | Stable |
| `P-G-DIST-01` | `Global Distance + Diversity Diagnostics` | `I-02`, `I-03` | Keep | Keep | Stable |
| `P-G-STATE-01` | `Fixed Reference Optima` + `Global Color Distribution Curves` + `Grid Snapshot` | `I-02`, `I-03` | Keep | Optional appendix | Keep, reduce prominence in thesis mode |
| `P-G-CTX-01` | `Static Overview` page (currently includes preference-order block) | `I-11` | Keep | Optional | Keep but prefer group-distribution style context over ordering block for readability |
| `P-G-CTX-02` | `Per-Area Personality Group Distributions` | `I-11` | Keep | Optional | Keep |
| `P-A-CORE-01` | `Area Color Distribution Curves` + `dist_to_reality` | `I-02`, `I-03` | Keep | Keep | Stable |
| `P-A-PUZ-01` | `Puzzle Distribution Curves` + `Puzzle Distance vs Outcome / Power` | `I-04` | Keep | Optional appendix | Stable |
| `P-A-PUZ-02` | `Decomposition A/B/C` (split page) | `I-04` | Keep | Optional appendix | Stable (split solved overload) |
| `P-A-VMODE-01` | `Rank-1 Match to Puzzle / Elected Outcome by Vote Mode` | `I-04`, `I-06` | Keep | Drop | Keep in debug only |
| `P-A-VMODE-02` | `Vote-mode Coverage` | `I-12` | Keep | Drop | Keep in debug only |
| `P-A-OPP-01` | `Group Opportunity Alignment to Puzzle` + spread diagnostics | `I-05` | Keep | Drop | Keep in debug only |
| `P-A-GATE-01` | `Puzzle Anti-Monopoly Gate` | `I-04`, `I-09` | Keep | Optional appendix | Keep |
| `P-A-GRP-01` | participants/eligible/non-altruistic + composition share | `I-06` | Keep | Drop | Keep, maybe compact legends |
| `P-A-GRP-02` | non-altruistic share + turnout by group | `I-06` | Keep | Optional | Keep |
| `P-A-GRP-03` | non-altruistic share distribution pages | `I-06` | Merge/compact | Drop | Merge into fewer pages |
| `P-A-GRP-04` | incentives/costs (`delta_rel`, fee) page | `I-07` | Keep | Drop | Keep as input context to causal learning page |
| `P-A-GRP-05` | learning direction + pressure (proxy-heavy) | `I-07` | Replace | Drop | Replace with causal decomposition (below) |
| `P-A-GRP-06` | participation shift pages (group / participant-abstainer / vote-mode split) | `I-07` | Merge | Drop | Merge into one causal page |
| `P-A-GRP-07` | dropout + vote-mode switching pages | `I-06` | Keep | Drop | Keep debug-only (small-group transparency already added) |
| `P-A-GRP-08` | within-group dispersion/inequality page | `I-06`, `I-12` | Optional | Drop | Keep as optional debug appendix |
| `P-A-GRP-09` | altruism signal + altruism shift | `I-08` | Keep | Optional | Keep |
| `P-A-MEAN-01` | `Gini Dissatisfaction` + `Mean Dissatisfaction by Group` | `I-01`, `I-06` | Keep | Keep (compact) | Stable |
| `P-A-MEAN-02` | `Gini Assets` + `Assets share by Group` | `I-01`, `I-06` | Keep | Keep (compact) | Stable |
| `P-A-REF-01` | `dist_to_ref_*` + `Area Mean Assets + Mean Dissatisfaction` | `I-03`, `I-01` | Keep | Keep | Keep, but avoid unreadable dual-scale spikes |
| `P-DOE-RUN-01` | `run_overview.pdf` (meta + knobs + personality group dists) | `I-11` | Keep | n/a | Stable (recently improved) |
| `P-DOE-AN-01` | `doe_analysis_summary.pdf` (correlations + effect/optimum plot) | `I-10` | Keep | n/a | Keep, later add uncertainty/confidence view |

---

## 4. Required New Debug Plot (P0)

### `P-A-LEARN-CAUSAL-01`: Participation Learning Causal Decomposition

This is the highest-priority missing plot family for `area_<id>.pdf` in `debug_doe`.

Goal:

- make participation learning dynamics directly auditable from one page,
- remove dependence on multiple proxy-heavy pages.

Must explicitly show (as requested):

- “average against all groups” reference (`mu_bar`)
- per-group average (`mu_g`)
- effective learning signal that updates participation
- split by group and by participants vs abstainers

#### Target page layout (single page, 4 panels)

1. **Relative-Performance Base**
   - per-group `mu_g = mean(election_delta_rel | eligible, group g, step t)`
   - global `mu_bar = mean(mu_g over eligible groups)` as dashed black reference
2. **Group Relative Component**
   - `r_g = w_g * (mu_g - mu_bar)`
   - secondary axis: `w_g = n_g / (n_g + participation_signal_group_shrink_k)`
3. **Signal Decomposition by Subpopulation**
   - participants: mean total signal, mean fee component, mean group component
   - abstainers: mean total signal, mean group component (fee term is zero by contract)
4. **Actual Update Effect**
   - exact `mean(Δq_participation)` for participants/abstainers
   - optional next-step turnout delta by group for response context

#### Required data-contract fields (group-step level)

Existing fields already useful:

- `participants_mean_participation_signal`
- `abstainers_mean_participation_signal`
- `participants_mean_participation_q_delta`
- `abstainers_mean_participation_q_delta`
- `participants_mean_fee_over_assets`

Add explicit causal fields for unambiguous interpretation:

- `group_mu_delta_rel`
- `global_mu_delta_rel`
- `group_signal_shrink_weight`
- `group_signal_component`
- `participants_mean_signal_group_component`
- `participants_mean_signal_fee_component`
- `abstainers_mean_signal_group_component`

#### Acceptance checks (must hold on debug page)

- In `group_relative_delta_rel_party`, `abstainers_mean_signal_total ~= group_signal_component`.
- In the same mode, `participants_mean_signal_total ~= group_signal_component + fee_component`.
- groups with residents `< 5` must remain visually de-emphasized.

---

## 5. Mode Targets (What We Should Ship)

### `debug_doe` target (current project phase)

Keep broad diagnostics, but organized around `I-01..I-12` and with fewer redundant pages:

- retain puzzle/power/gate pages
- retain group behavior pages with small-group transparency
- replace multi-page learning proxies with `P-A-LEARN-CAUSAL-01`
- keep packet-level DOE context (`run_overview`, `doe_analysis_summary`, queue)

### `thesis_run` target (later)

Compact, confirmatory-first set:

- core global outcomes (`I-01`)
- global/area quality (`I-02`, `I-03`)
- compact group means context (`I-06`)
- optional appendix pages for puzzle/power and anti-monopoly (`I-04`) only when needed

No debug-only page families in primary thesis packet.

---

## 6. Execution Plan (DOE/Debug First, Feedback-Loop Friendly)

1. Lock this insight inventory as the decision baseline.
2. Implement `P-A-LEARN-CAUSAL-01` data fields and page.
3. Merge/remove redundant learning proxy pages (`P-A-GRP-05`, `P-A-GRP-06`) after parity checks.
4. Keep existing small-group transparency behavior and extend to any remaining noisy group plots if needed.
5. Run DOE packet validation on representative top/mid/bottom samples.
6. Start plot-by-plot documentation pass:
   - one canonical plot definition section per plot ID,
   - PDF docs reference plot IDs (no duplicate explanations).
7. After debug mode stabilizes, implement strict `thesis_run` profile pruning.

---

## 7. Notes

- This inventory intentionally does **not** treat every existing plot as equally valid.
- If a current plot cannot be tied to an insight ID, it should be removed or moved to optional debug appendix.
- If an insight ID has weak/fragmented coverage, new plots or data fields are preferred over adding more proxy variants.
