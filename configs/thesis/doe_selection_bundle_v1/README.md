# DOE Selection Bundle v1

This lightweight bundle is the published audit packet for the locked DOE design selection.

Purpose:

- allow external re-scoring and audit of the selected design (`design_id=149`),
- avoid publishing heavy raw DOE per-run directories.

The bundle is intentionally limited to selection-level artifacts and features.

Primary verifier/audit commands:

```bash
PYTHONPATH=. python scripts/repro/verify_thesis_repro_bundle.py
PYTHONPATH=. python scripts/repro/audit_doe_design_lock.py
```
