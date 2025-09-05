"""
Build an execution run plan from best_pairs_summary.json to drive targeted backtests.

- Picks top-K pairs per strategy (filters optional by min PF / max DD)
- Resolves full strategy file paths across strategy folders
- Emits a run plan JSON with tasks per (strategy, symbol, timeframe) including
  suggested risk settings when available

Usage:
  python Backtesting/tool_box/operations/build_run_plan.py [--top 3] [--min_pf 1.3] [--max_dd 40]
                                                            [--summary <path_to_best_pairs_summary.json>]
Output:
  Backtesting/results/winners/run_plan_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import os
import re
import json
import glob
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional


PROJECT_ROOT = r"C:\Users\andre\OneDrive\Desktop\Mastercode"
STRATEGIES_DIR = os.path.join(PROJECT_ROOT, "Backtesting", "strategies")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Backtesting", "results")
WINNERS_DIR = os.path.join(RESULTS_DIR, "winners")


def _latest_best_pairs_summary() -> Optional[str]:
    if not os.path.isdir(WINNERS_DIR):
        return None
    files = [
        os.path.join(WINNERS_DIR, f)
        for f in os.listdir(WINNERS_DIR)
        if f.startswith("best_pairs_summary_") and f.endswith(".json")
    ]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _find_strategy_path(name: str) -> Optional[str]:
    # Search common subfolders for the given strategy filename
    patterns = [
        os.path.join(STRATEGIES_DIR, "**", name),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            # Prefer non-broken, non-old_winners if multiple
            matches.sort(key=lambda p: ("broken" in p, "old_winners" in p, len(p)))
            return matches[0]
    return None


def build_run_plan(summary_path: str, top_k: int, min_pf: float, max_abs_dd: float) -> Dict[str, Any]:
    with open(summary_path, "r") as f:
        payload = json.load(f)

    strategies = payload.get("strategies", {})
    # When produced by best_pairs_report, structure is { strategies: { <strategy>: { top_pairs: [ PairAggregate... ] } } }
    # If a different shape is detected, try to adapt
    run_tasks: List[Dict[str, Any]] = []

    if isinstance(strategies, dict):
        iterable = strategies.items()
    elif isinstance(strategies, list):
        # fall back: treat list like winners summary; no pairs info
        iterable = [(s.get("strategy", "UNKNOWN"), {"top_pairs": []}) for s in strategies]
    else:
        iterable = []

    for strategy_name, info in iterable:
        top_pairs = info.get("top_pairs", []) if isinstance(info, dict) else []
        # filter pairs by thresholds
        filtered = []
        for p in top_pairs:
            pf = float(p.get("median_profit_factor", 0.0))
            dd = abs(float(p.get("median_max_drawdown_pct", 0.0)))
            if pf >= min_pf and dd <= max_abs_dd:
                filtered.append(p)
        selected = filtered[: top_k] if filtered else top_pairs[: top_k]

        for p in selected:
            symbol = p.get("symbol")
            timeframe = p.get("timeframe")
            settings = p.get("representative_settings")
            strategy_path = _find_strategy_path(strategy_name)
            if not strategy_path:
                # Try exact filename fallback
                strategy_path = _find_strategy_path(os.path.basename(strategy_name))

            run_tasks.append({
                "strategy": strategy_name,
                "strategy_path": strategy_path,
                "symbol": symbol,
                "timeframe": timeframe,
                "median_profit_factor": p.get("median_profit_factor"),
                "median_return_pct": p.get("median_return_pct"),
                "median_max_drawdown_pct": p.get("median_max_drawdown_pct"),
                "engines_covered": p.get("engines_covered", []),
                "suggested_risk_settings": settings or {},
            })

    plan = {
        "generated_at": datetime.now().isoformat(),
        "source_summary": summary_path,
        "selection": {
            "top_k_per_strategy": top_k,
            "min_profit_factor": min_pf,
            "max_abs_drawdown_pct": max_abs_dd,
        },
        "tasks": run_tasks,
    }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted backtest run plan from best pairs summary")
    parser.add_argument("--top", type=int, default=3, help="Top K pairs per strategy")
    parser.add_argument("--min_pf", type=float, default=1.3, help="Minimum median profit factor filter")
    parser.add_argument("--max_dd", type=float, default=45.0, help="Maximum absolute median drawdown percent")
    parser.add_argument("--summary", type=str, default=None, help="Path to best_pairs_summary JSON")
    args = parser.parse_args()

    summary_path = args.summary or _latest_best_pairs_summary()
    if not summary_path or not os.path.exists(summary_path):
        print("No best_pairs_summary JSON found. Run best_pairs_report first.")
        return

    plan = build_run_plan(summary_path, args.top, args.min_pf, args.max_dd)
    os.makedirs(WINNERS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(WINNERS_DIR, f"run_plan_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Run plan written: {out_path}")


if __name__ == "__main__":
    main()


