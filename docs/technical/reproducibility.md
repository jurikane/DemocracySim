# Reproducibility

This thesis follows a **two-tier reproducibility contract**.

## Scope

### Tier 1: Reproducing the final thesis runs

The final thesis run setup is fully specified in the repository via:

- `configs/thesis/final_model_v1.yaml`
- `configs/thesis/final_seed_list_v1.json`
- `configs/thesis/final_run_manifest_v1.csv`
- `configs/thesis/freeze_provenance_v1.json`

Frozen run matrix:

- main family: `5 rules x 200 matched seeds = 1000 runs`
- approval context arm: `50 runs`
- total: `1050 runs`

### Tier 2: Auditing the DOE design lock

The DOE-based design lock (`design_id=149`) can be audited from the compact bundle:

- `configs/thesis/doe_selection_bundle_v1/`
- `configs/thesis/doe_selection_provenance_v1.json`

The full raw DOE directory (≈30 GB) is not required for this audit.

## What is intentionally not published

The raw DOE directory with all per-run artifacts is intentionally not included.

Rationale:

- size and hosting practicality,
- the thesis requires reproducible **selection logic** and **final runs**, not archival of all intermediate run folders.

## Environment

Tested with:

- Python `3.11.9`
- pinned runtime dependencies in `requirements.txt`
- freeze snapshot in `requirements-lock.txt`

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To more closely match the freeze environment:

```bash
pip install -r requirements-lock.txt
```

## Verification

Verify the public reproducibility bundle:

```bash
PYTHONPATH=. python scripts/repro/verify_thesis_repro_bundle.py
```

This checks:

- artifact hash block consistency,
- manifest structure and counts,
- deterministic regeneration of the `final_run_manifest_v1.csv` hash.

Audit the DOE design lock:

```bash
PYTHONPATH=. python scripts/repro/audit_doe_design_lock.py
```

This checks:

- ranking recomputation from `doe_run_features.csv` under the locked objective contract,
- top design equals `design_id=149`,
- score consistency with `doe_design_scores.csv`.

## Running the final manifest

Dry-run:

```bash
PYTHONPATH=. python scripts/run_final_manifest.py --dry-run
```

Full execution:

```bash
PYTHONPATH=. python scripts/run_final_manifest.py
```

Resume:

```bash
PYTHONPATH=. python scripts/run_final_manifest.py --resume
```

## Stamping provenance

For the thesis freeze, provenance was stamped from a clean commit using:

```bash
PYTHONPATH=. python scripts/repro/restamp_freeze_provenance.py --release-tag thesis-freeze-v1 --release-url <release-url>
```

This records git_head, git_dirty, release metadata, and the artifact hash block.
