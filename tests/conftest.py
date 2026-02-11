"""Test harness configuration.

These tests are often run in headless environments (CI, agents). Ensure
Matplotlib uses a non-interactive backend so plotting code doesn't abort.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Avoid Matplotlib trying to write under ~/.matplotlib in restricted envs.
    cfg_dir = Path(tempfile.gettempdir()) / "demosim-matplotlib"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg_dir))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        # If matplotlib isn't installed for some reason, tests that depend on it
        # will fail normally; we don't want conftest itself to crash.
        pass
