"""
Build a consolidated winners Markdown report from winners_summary.json.

Usage:
  python Backtesting/tool_box/reports/build_winners_markdown.py <path_to_winners_summary.json>
If no path provided, picks the latest file under Backtesting/results/winners/.
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from typing import Any, Dict

PROJECT_ROOT = r"C:\Users\andre\OneDrive\Desktop\Mastercode"
WINNERS_DIR = os.path.join(PROJECT_ROOT, "Backtesting", "results", "winners")


def _latest_json_in(dir_path: str) -> str | None:
    if not os.path.isdir(dir_path):
        return None
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.json') and f.startswith('winners_summary_')
    ]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def build_md(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Generated: {payload.get('generated_at', datetime.now().isoformat())}")
    lines.append("")
    lines.append("Top Strategies (ranked, excluding Sharpe):")
    lines.append("")
    lines.append("| Rank | Strategy | Score | PF (med) | Return% (med) | MaxDD% (med) | Win% (med) | Profitable% | WF+Win% (med) | Engines |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    strategies = payload.get('strategies') or []
    for idx, s in enumerate(strategies, start=1):
        lines.append(
            "| {rank} | {name} | {score:.2f} | {pf:.2f} | {ret:.2f} | {dd:.2f} | {wr:.2f} | {prof:.1f}% | {wfpos:.1f}% | {engs} |".format(
                rank=idx,
                name=s.get('strategy', 'UNKNOWN'),
                score=float(s.get('score', 0.0)),
                pf=float(s.get('median_profit_factor', 0.0)),
                ret=float(s.get('median_return_pct', 0.0)),
                dd=float(s.get('median_max_drawdown_pct', 0.0)),
                wr=float(s.get('median_win_rate_pct', 0.0)),
                prof=float(s.get('percent_profitable_evaluations', 0.0)),
                wfpos=float(s.get('median_wf_positive_window_ratio', 0.0)*100.0),
                engs=",".join(s.get('engines_covered', [])),
            )
        )
    return "\n".join(lines)


def main() -> None:
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    path = arg_path or _latest_json_in(WINNERS_DIR)
    if not path or not os.path.exists(path):
        print("No winners_summary JSON found.")
        return
    with open(path, 'r') as f:
        payload = json.load(f)
    md = build_md(payload)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(WINNERS_DIR, f"winners_compiled_{ts}.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Winners markdown written: {out_path}")


if __name__ == "__main__":
    main()


