"""
Utility: Sort winners/losers from basic backtest results and populate Data/winners.

- Scans results/basic_backtest_* for per-strategy winners JSONs
- Maps basenames to CSVs in Data/Hyperliquid (or other pools) and copies into
  Data/winners/<strategy>/, creating subfolders as needed
"""

import os
import json
import shutil
from pathlib import Path
from typing import List

RESULTS_ROOT = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results")
WINNERS_ROOT = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data\winners")
DATA_POOLS = [
    Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data\Hyperliquid"),
]

def find_latest_basic_results() -> Path:
    candidates = [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and d.name.startswith("basic_backtest_")]
    if not candidates:
        raise SystemExit("No basic_backtest_* result directories found")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_csv_in_pools(basename: str) -> Path:
    for pool in DATA_POOLS:
        candidate = None
        # scan recursively
        for path in pool.rglob(f"{basename}.csv"):
            candidate = path
            break
        if candidate is not None:
            return candidate
    raise FileNotFoundError(f"CSV for {basename} not found in data pools")

def copy_winner(strategy: str, basename: str):
    src = find_csv_in_pools(basename)
    dst_dir = WINNERS_ROOT / strategy
    ensure_dir(dst_dir)
    dst = dst_dir / f"{basename}.csv"
    if not dst.exists():
        shutil.copy2(src, dst)

def collect_winners_from_dir(basic_dir: Path):
    for strat_dir in basic_dir.iterdir():
        if not strat_dir.is_dir():
            continue
        winners = strat_dir / 'winners'
        if not winners.exists():
            continue
        for f in winners.glob('*.json'):
            basename = f.stem
            copy_winner(strat_dir.name, basename)

def main():
    latest = find_latest_basic_results()
    collect_winners_from_dir(latest)
    print(f"Winners copied from {latest}")

if __name__ == "__main__":
    main()


