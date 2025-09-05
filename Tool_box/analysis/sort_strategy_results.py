"""
Sort per-strategy result JSONs into winners/losers subfolders.

Rules:
- For each strategy directory under the selected results run, create
  `winners/` and `losers/` subfolders.
- Any result with return < 10% goes to `losers/`; otherwise to `winners/`.
- Operates per-strategy; does NOT aggregate into a central winners/losers.

Usage:
  # Sort a specific run
  python sort_strategy_results.py \
    --results-dir "C:/.../Backtesting/results/basic_backtest_YYYYMMDD_HHMMSS" \
    --threshold 10

  # Sort ALL runs under Backtesting/results
  python sort_strategy_results.py --all-runs --threshold 10

If neither --results-dir nor --all-runs is provided, the latest engine
run directory within Backtesting/results is used automatically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple


RETURN_KEYS = [
    "return_pct",  # preferred (percent scale)
    "Return [%]",  # raw stats label
    "total_return",  # sometimes fraction scale
    "total_return_pct",
]


def find_backtesting_root(start: Path) -> Path:
    """Find the Backtesting directory from any starting path."""
    for parent in [start] + list(start.parents):
        if (parent / "Backtesting" / "results").exists():
            return parent / "Backtesting"
    # Fallback: assume current file is within Backtesting/tool_box
    return start


def pick_latest_results_directory(results_root: Path) -> Optional[Path]:
    candidates = [
        p for p in results_root.iterdir() if p.is_dir() and p.name.lower() != "logs"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_json_return(file_path: Path) -> Optional[float]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # Try multiple keys
    for key in RETURN_KEYS:
        if key in data:
            val = data[key]
            try:
                return float(val)
            except Exception:
                continue

    # Some engines may nest metrics under 'performance_summary' or similar
    for container_key in ["performance_summary", "performance", "stats"]:
        if container_key in data and isinstance(data[container_key], dict):
            for key in RETURN_KEYS:
                if key in data[container_key]:
                    try:
                        return float(data[container_key][key])
                    except Exception:
                        continue

    return None


def to_percent_scale(value: float) -> float:
    """Normalize return to percent scale.

    Heuristic:
    - If abs(value) <= 1.0 → assume fraction (e.g., 0.12) ⇒ convert to percent (12.0)
    - Else assume already percent (e.g., 12.0)
    """
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v * 100.0 if -1.0 <= v <= 1.0 else v


def sort_strategy_folder(strategy_dir: Path, threshold_pct: float, dry_run: bool = False) -> Tuple[int, int, int]:
    """Sort JSON files under a single strategy folder into winners/losers.

    Returns: (moved_winners, moved_losers, skipped)
    """
    winners_dir = strategy_dir / "winners"
    losers_dir = strategy_dir / "losers"
    if not dry_run:
        winners_dir.mkdir(exist_ok=True)
        losers_dir.mkdir(exist_ok=True)

    moved_winners = moved_losers = skipped = 0

    for entry in strategy_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".json":
            continue

        # Skip non-result aggregates by name if needed
        if entry.name in {"all_results.json", "basic_backtest_summary.json", "alpha_analysis_summary.json", "risk_management_summary.json", "performance_evaluation_summary.json"}:
            skipped += 1
            continue

        ret_val = read_json_return(entry)
        ret_pct = to_percent_scale(ret_val) if ret_val is not None else float("nan")

        is_winner = False
        if ret_pct == ret_pct:  # not NaN
            is_winner = ret_pct >= threshold_pct

        target_dir = winners_dir if is_winner else losers_dir
        target_path = target_dir / entry.name

        if dry_run:
            if is_winner:
                moved_winners += 1
            else:
                moved_losers += 1
        else:
            try:
                entry.replace(target_path)
                if is_winner:
                    moved_winners += 1
                else:
                    moved_losers += 1
            except Exception:
                skipped += 1

    return moved_winners, moved_losers, skipped


def main():
    parser = argparse.ArgumentParser(description="Sort per-strategy results into winners/losers.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to a specific engine results directory (e.g., .../results/basic_backtest_YYYYMMDD_HHMMSS). If omitted, the latest will be used.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all engine run directories under Backtesting/results.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Winner threshold in percent (default: 10.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without changing files.",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step through one strategy folder at a time with confirmation prompts.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run without prompts (overrides --step).",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    backtesting_root = find_backtesting_root(script_path)
    results_root = backtesting_root / "results"

    # Build list of run directories to process
    run_dirs = []
    if args.results_dir:
        candidate = Path(args.results_dir)
        if candidate.exists() and candidate.is_dir():
            run_dirs = [candidate]
    elif args.all_runs:
        run_dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.lower() != "logs"]
        run_dirs.sort(key=lambda p: p.stat().st_mtime)
    else:
        latest = pick_latest_results_directory(results_root)
        if latest is not None:
            run_dirs = [latest]

    if not run_dirs:
        print("❌ No results directories found. Use --results-dir or --all-runs.")
        return

    # Default to step mode if no explicit scope flags provided
    step_mode = args.step
    if args.auto:
        step_mode = False
    elif not args.results_dir and not args.all_runs:
        # No explicit scope supplied → assume interactive stepping by default
        step_mode = True

    grand_winners = grand_losers = grand_skipped = 0

    for engine_dir in run_dirs:
        print(f"\n📂 Sorting results in: {engine_dir}")
        total_winners = total_losers = total_skipped = 0

        # Each subdirectory is expected to be a strategy folder.
        for strategy_dir in engine_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            if strategy_dir.name.lower() in {"logs", "aggregated", "visualizations"}:
                continue

            if step_mode and not args.dry_run:
                # Preview
                pw, pl, ps = sort_strategy_folder(strategy_dir, args.threshold, dry_run=True)
                user = input(
                    f"  - Preview {strategy_dir.name}: winners={pw}, losers={pl}, skipped={ps}. Press Enter to sort, 's' to skip, 'q' to quit: "
                ).strip().lower()
                if user == 'q':
                    print("Stopping at user request.")
                    print(f"Totals for {engine_dir.name} → winners={total_winners}, losers={total_losers}, skipped={total_skipped}")
                    print("\n✅ All runs processed.")
                    print(f"Grand totals → winners={grand_winners}, losers={grand_losers}, skipped={grand_skipped}")
                    return
                if user == 's':
                    print(f"  - Skipped {strategy_dir.name}")
                    continue
                # Proceed to execute
                w, l, s = sort_strategy_folder(strategy_dir, args.threshold, dry_run=False)
            else:
                w, l, s = sort_strategy_folder(strategy_dir, args.threshold, dry_run=args.dry_run)

            total_winners += w
            total_losers += l
            total_skipped += s
            print(f"  - {strategy_dir.name}: winners={w}, losers={l}, skipped={s}")

        print(f"Totals for {engine_dir.name} → winners={total_winners}, losers={total_losers}, skipped={total_skipped}")
        grand_winners += total_winners
        grand_losers += total_losers
        grand_skipped += total_skipped

    print("\n✅ All runs processed.")
    print(f"Grand totals → winners={grand_winners}, losers={grand_losers}, skipped={grand_skipped}")


if __name__ == "__main__":
    main()


