"""
CLI to aggregate a single backtesting run's results into a compact report.

Usage:
  python Backtesting/tool_box/aggregate_results.py \
    --run "C:/.../Backtesting/results/basic_backtest_YYYYMMDD_HHMMSS"

If --run is omitted, the latest run under Backtesting/results is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from result_analyzer import collect_results, save_aggregates


def find_backtesting_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "Backtesting" / "results").exists():
            return parent / "Backtesting"
    return start


def pick_latest_results_directory(results_root: Path) -> Optional[Path]:
    candidates = [
        p for p in results_root.iterdir() if p.is_dir() and p.name.lower() != "logs"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate backtesting run results")
    parser.add_argument("--run", type=str, default=None, help="Path to a run directory")
    args = parser.parse_args()

    script = Path(__file__).resolve()
    backtesting_root = find_backtesting_root(script)
    results_root = backtesting_root / "results"

    run_dir = Path(args.run) if args.run else pick_latest_results_directory(results_root)
    if run_dir is None or not run_dir.exists():
        print("❌ Run directory not found. Use --run to specify one.")
        return

    print(f"📦 Aggregating results in: {run_dir}")
    rows = collect_results(run_dir)
    save_aggregates(run_dir, rows)
    print("✅ Aggregation completed.")


if __name__ == "__main__":
    main()


