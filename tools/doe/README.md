# DOE Tools

Utilities for DOE-specific analysis and review workflows.

## Scripts

- `build_doe_hil_queue.py`: build top/mid/bottom HIL queue from scored DOE.
- `doe_hil_review.py`: inspect queue rows, print/run summary commands, populate AI notes.
- `build_doe_review_bundle.py`: assemble a compact review bundle from DOE outputs.
- `probe_scoring_dimensions.py`: stress-test scoring dimensions by dominant-weight probing.
- `recovery_scan.py`: scan run-level artifacts for strict/moderate recovery patterns.
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
