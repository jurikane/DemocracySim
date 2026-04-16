# DOE Tools

Utilities for DOE-specific analysis and review workflows.

## Scripts

- `build_doe_review_bundle.py`: assemble a compact review bundle from DOE outputs.
- `probe_scoring_dimensions.py`: stress-test scoring dimensions by dominant-weight probing.
- `doe_inference.py`: generate DOE inference artifacts (seed effects, nonlinear importance, interactions, CIs, Pareto).
- `select_balanced_seeds.py`: choose stratified candidate seeds from DOE profile descriptors.
- `run_sanity_matrix.py`: execute a compact non-DOE sanity scenario matrix.

## Usage

Run via module path, for example:

```bash
python -m tools.doe.probe_scoring_dimensions --help
```

## Contract

- Inputs/outputs should be explicit and file-based.
- Keep scripts thin; reusable logic belongs in `src/analysis` or other `src/*` modules.
