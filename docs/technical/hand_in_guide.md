# Hand-In Guide

This repository distinguishes between the cited scientific freeze and the curated submission snapshot:

- cited thesis freeze: branch `thesis`, tag `thesis-freeze-v1`
- curated USB hand-in snapshot: branch `thesis-handin`, tag `thesis-handin-v1`

The thesis text cites the first freeze. The second snapshot is only for submission packaging and USB verification.

Legacy frozen manifest and copied final-run directory labels are preserved for compatibility with the cited freeze.

## Verify First

Create a fresh Python `3.11.9` environment and install runtime dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the verification-first sequence from the clean hand-in checkout:

```bash
PYTHONPATH=. python scripts/repro/verify_thesis_repro_bundle.py
PYTHONPATH=. python scripts/repro/audit_doe_design_lock.py
PYTHONPATH=. python scripts/run_final_manifest.py --dry-run
PYTHONPATH=. python scripts/build_thesis_analysis_package.py --dry-run
```

Expected hand-in payload inside the clean working tree:

- repository checkout at tag `thesis-handin-v1`
- `data/simulation_output/thesis_final_runs_v1`
- `artifacts/thesis_analysis_v1`

USB-side extras outside the repo checkout:

- `README.md` at the top level of the USB folder
- `SHA256SUMS.txt` at the top level of the USB folder
- `thesis_kuehnel.pdf` at the top level of the USB folder

To keep the clean checkout readable without changing tracked ignore rules, add local excludes for `data/` and `artifacts/` in `.git/info/exclude`.

## Optional Rerun From Manifest

The hand-in is verification-first. Full regeneration is optional and should only be done after the checks above pass.

Re-run the frozen final manifest:

```bash
PYTHONPATH=. python scripts/run_final_manifest.py
```

Rebuild the thesis analysis package:

```bash
PYTHONPATH=. python scripts/build_thesis_analysis_package.py
```

Rebuild the appendix-support free-rider support table:

```bash
PYTHONPATH=. python scripts/build_free_rider_support_csv.py
```

The appendix-support tables `free-rider.csv` and `free-rider-support.csv` are descriptive support outputs shipped with the hand-in package alongside the core `T1` to `T6` and `F1` to `F4` deliverables.
