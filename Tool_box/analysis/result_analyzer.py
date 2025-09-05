"""
Result aggregation utilities for Backtesting engine runs.

Scans one results run directory (e.g., Backtesting/results/basic_backtest_YYYYMMDD_HHMMSS)
and aggregates metrics across all strategy folders.

Outputs:
- aggregated/all_results.json: flat list of all parsed result rows
- aggregated/summary.json: high-level summary stats
- aggregated/all_results.csv: tabular results
- aggregated/pivots/*.csv: helpful pivots (avg_return_by_symbol_timeframe, avg_sharpe_by_symbol_timeframe)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ParsedResult:
    strategy: str
    data_file: str
    symbol: str
    timeframe: str
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    total_trades: int
    path: str


def _safe_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key, default)
        return float(v) if v is not None else default
    except Exception:
        return default


def _safe_int(d: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        v = d.get(key, default)
        return int(v) if v is not None else default
    except Exception:
        return default


def parse_result_json(file_path: Path) -> Optional[ParsedResult]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # Prefer top-level fields used by current engines
    strategy = str(data.get("strategy") or data.get("strategy_name") or "unknown")
    data_file = str(data.get("data_file") or os.path.splitext(file_path.name)[0])
    symbol = str(data.get("symbol") or "unknown")
    timeframe = str(data.get("timeframe") or "unknown")

    # Numeric metrics
    return_pct = _safe_float(data, "return_pct", _safe_float(data, "Return [%]", 0.0))
    sharpe = _safe_float(data, "sharpe_ratio", _safe_float(data, "Sharpe Ratio", 0.0))
    mdd = _safe_float(data, "max_drawdown", _safe_float(data, "Max. Drawdown [%]", 0.0))
    pf = _safe_float(data, "profit_factor", _safe_float(data, "Profit Factor", 0.0))
    win_rate = _safe_float(data, "win_rate", _safe_float(data, "Win Rate [%]", 0.0))
    total_trades = _safe_int(data, "total_trades", _safe_int(data, "# Trades", 0))

    return ParsedResult(
        strategy=strategy,
        data_file=data_file,
        symbol=symbol,
        timeframe=timeframe,
        return_pct=return_pct,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        profit_factor=pf,
        win_rate=win_rate,
        total_trades=total_trades,
        path=str(file_path),
    )


def collect_results(run_dir: Path) -> List[ParsedResult]:
    rows: List[ParsedResult] = []
    for strat_dir in run_dir.iterdir():
        if not strat_dir.is_dir():
            continue
        lname = strat_dir.name.lower()
        if lname in {"logs", "aggregated", "visualizations"}:
            continue

        for entry in strat_dir.iterdir():
            if not entry.is_file() or entry.suffix.lower() != ".json":
                continue
            # Skip known aggregate files
            if entry.name in {
                "all_results.json",
                "basic_backtest_summary.json",
                "alpha_analysis_summary.json",
                "risk_management_summary.json",
                "performance_evaluation_summary.json",
                "statistical_summary.json",
            }:
                continue
            parsed = parse_result_json(entry)
            if parsed:
                rows.append(parsed)
    return rows


def save_aggregates(run_dir: Path, rows: List[ParsedResult]) -> None:
    if not rows:
        return
    aggregated_dir = run_dir / "aggregated"
    pivots_dir = aggregated_dir / "pivots"
    aggregated_dir.mkdir(exist_ok=True)
    pivots_dir.mkdir(exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame([r.__dict__ for r in rows])

    # Save all rows
    (aggregated_dir / "all_results.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    df.to_csv(aggregated_dir / "all_results.csv", index=False)

    # Summary
    summary = {
        "total_results": int(len(df)),
        "strategies": int(df["strategy"].nunique()),
        "symbols": int(df["symbol"].nunique()),
        "timeframes": int(df["timeframe"].nunique()),
        "avg_return_pct": float(df["return_pct"].mean()),
        "avg_sharpe": float(df["sharpe_ratio"].mean()),
        "avg_max_drawdown": float(df["max_drawdown"].mean()),
        "avg_profit_factor": float(df["profit_factor"].mean()),
        "avg_win_rate": float(df["win_rate"].mean()),
        "total_trades": int(df["total_trades"].sum()),
    }
    (aggregated_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Helpful pivots
    def safe_pivot(value_col: str, fname: str) -> None:
        try:
            pivot = (
                df.pivot_table(
                    index="symbol", columns="timeframe", values=value_col, aggfunc="mean"
                )
                .round(4)
                .sort_index()
            )
            pivot.to_csv(pivots_dir / fname)
        except Exception:
            pass

    safe_pivot("return_pct", "avg_return_by_symbol_timeframe.csv")
    safe_pivot("sharpe_ratio", "avg_sharpe_by_symbol_timeframe.csv")


__all__ = [
    "ParsedResult",
    "collect_results",
    "save_aggregates",
]


