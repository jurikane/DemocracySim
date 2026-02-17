from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.analysis.summary_tooling import (
    generate_run_summary_batch2,
)
from src.utils.run_path_picker import (
    normalize_selected_run_dir,
    pick_run_dir_interactive,
    resolve_run_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis summary sidecars + run-level PDFs from one run directory."
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a single run directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for summary artifacts (default: <run-dir>/analysis)",
    )
    parser.add_argument(
        "--closed",
        action="store_true",
        help="Do not auto-open generated summary PDFs.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "full"),
        default="full",
        help="Summary compute mode: fast skips expensive benchmark optimization; full computes all references.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable benchmark reference cache reuse.",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir) if args.run_dir is not None else pick_run_dir_interactive(action_label="summarize")
    if run_dir is None:
        print("No run selected. Usage: python -m scripts.generate_summary --run-dir <run_dir>")
        return

    normalized = normalize_selected_run_dir(run_dir, action_label="summarize")
    if normalized is None:
        return
    run_dir = normalized

    if not run_dir.exists():
        print("Run directory does not exist:", run_dir)
        return

    artifacts = generate_run_summary_batch2(
        run_dir=run_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        use_cache=not args.no_cache,
    )
    print(f"Wrote: {artifacts.global_series_csv}")
    print(f"Wrote: {artifacts.area_series_csv}")
    print(f"Wrote: {artifacts.summary_stats_json}")
    if artifacts.static_overview_pdf is not None:
        print(f"Wrote: {artifacts.static_overview_pdf}")
    if artifacts.global_summary_pdf is not None:
        print(f"Wrote: {artifacts.global_summary_pdf}")

    if not args.closed:
        _open_folder(artifacts.out_dir)


def _open_folder(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
            return
        if sys.platform.startswith("win"):
            # type: ignore[attr-defined]
            os_start = getattr(__import__("os"), "startfile", None)
            if os_start is not None:
                os_start(str(path))
                return
        subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        # Best-effort UX helper; generation artifacts are already written.
        return


if __name__ == "__main__":
    main()
