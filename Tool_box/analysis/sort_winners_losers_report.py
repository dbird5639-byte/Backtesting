"""
Generate a summary report of winners/losers per strategy from a basic_backtest_* folder.
"""

import os
from pathlib import Path
from typing import Dict

RESULTS_ROOT = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results")

def find_latest_basic_results() -> Path:
    candidates = [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and d.name.startswith("basic_backtest_")]
    if not candidates:
        raise SystemExit("No basic_backtest_* result directories found")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def build_report(basic_dir: Path) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {}
    for strat_dir in basic_dir.iterdir():
        if not strat_dir.is_dir():
            continue
        winners = strat_dir / 'winners'
        losers = strat_dir / 'losers'
        win_count = len(list(winners.glob('*.json'))) if winners.exists() else 0
        lose_count = len(list(losers.glob('*.json'))) if losers.exists() else 0
        report[strat_dir.name] = {'winners': win_count, 'losers': lose_count}
    return report

def main():
    latest = find_latest_basic_results()
    rep = build_report(latest)
    print("Strategy, Winners, Losers")
    for strat, counts in sorted(rep.items()):
        print(f"{strat}, {counts['winners']}, {counts['losers']}")

if __name__ == "__main__":
    main()


