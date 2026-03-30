# Final Analysis Package

The thesis-final analysis package is the curated derived output layer built from the frozen final run batch. It contains the figures, tables, and machine-readable summaries used for the evaluation chapter and for the university USB hand-in.

The repository tracks the build code, documentation, and tests for this package. The generated package payload itself is copied into the hand-in working tree under `artifacts/thesis_analysis_v1/`, while the raw final-run directories remain outside Git because of their size.

Package inventory:

- `derived/` for machine-readable analysis products
- `tables/` for thesis-ready tables
- `figures/` for thesis-ready figures
- `analysis_provenance.json` for package-level provenance and settings

Core package contents:

- `derived/run_level_endpoint_summary.csv`
- `derived/rule_step_primary_summary.csv`
- `derived/canonical_pairwise_effects.csv`
- `derived/reference_pairwise_effects.csv`
- `derived/robustness_alternative_readouts.csv`
- `tables/T1_frozen_run_protocol_provenance.csv` through `tables/T6_robustness_summaries.csv`
- `figures/F1_primary_metric_trajectories.(png|pdf)` through `figures/F4_reference_family_effect_panel.(png|pdf)`

Appendix-support tables shipped with the hand-in package:

- `tables/free-rider.csv`
- `tables/free-rider-support.csv`

For freeze artifacts, hashes, and final-run reproducibility details, see [Reproducibility](reproducibility.md). For the verification-first submission workflow, see [Hand-In Guide](hand_in_guide.md).
