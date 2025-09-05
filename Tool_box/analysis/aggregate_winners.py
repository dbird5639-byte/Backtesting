"""
Aggregate backtest result files and identify winning strategies.

Rules (user preference):
- Do NOT use Sharpe as a validation metric.
- Favor strategies with strong Profit Factor, positive/robust returns,
  acceptable drawdowns, and consistent performance across walkforward windows.

Outputs:
- winners_summary.json: machine-readable per-strategy aggregates and rankings
- winners_summary.md: human-readable ranked report

Usage (from project root):
  python Backtesting/tool_box/analysis/aggregate_winners.py
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


RESULTS_ROOT = os.path.join(
    r"C:\Users\andre\OneDrive\Desktop\Mastercode", "Backtesting", "Results"
)


def _is_finite_number(value: Any) -> bool:
    try:
        f = float(value)
        return math.isfinite(f)
    except Exception:
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _median(values: List[float], default: float = 0.0) -> float:
    finite = sorted(v for v in values if _is_finite_number(v))
    n = len(finite)
    if n == 0:
        return default
    mid = n // 2
    if n % 2 == 1:
        return float(finite[mid])
    return float((finite[mid - 1] + finite[mid]) / 2.0)


def _mean(values: List[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if _is_finite_number(v)]
    if not finite:
        return default
    return float(sum(finite) / len(finite))


def discover_json_files(results_root: str) -> List[str]:
    json_files: List[str] = []
    for root, dirs, files in os.walk(results_root):
        # Skip logs directory
        if os.path.basename(root) == "logs":
            continue
        for name in files:
            if not name.endswith(".json"):
                continue
            if name.endswith(".tmp"):
                continue
            json_files.append(os.path.join(root, name))
    return json_files


@dataclass
class NormalizedEvaluation:
    strategy: str
    engine: str
    data_file: str
    symbol: Optional[str]
    timeframe: Optional[str]

    # Core metrics for decisioning (Sharpe intentionally excluded)
    profit_factor: Optional[float]
    return_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    win_rate_pct: Optional[float]
    buy_hold_return_pct: Optional[float]
    total_trades: Optional[int]
    exposure_time_pct: Optional[float]

    # Walkforward performance consistency (when present)
    wf_avg_return_pct: Optional[float]
    wf_avg_profit_factor: Optional[float]
    wf_positive_window_ratio: Optional[float]
    wf_performance_consistency: Optional[float]


def _parse_single_result(data: Dict[str, Any], path: str, engine: str) -> Optional[NormalizedEvaluation]:
    strategy = data.get("strategy") or "UNKNOWN"
    data_file = data.get("data_file") or os.path.splitext(os.path.basename(path))[0]
    symbol = data.get("symbol")
    timeframe = data.get("timeframe")

    # Default metrics
    profit_factor = None
    return_pct = None
    max_drawdown_pct = None
    win_rate_pct = None
    buy_hold_return_pct = None
    total_trades = None
    exposure_time_pct = None

    wf_avg_return_pct = None
    wf_avg_profit_factor = None
    wf_positive_window_ratio = None
    wf_performance_consistency = None

    # Risk managed or basic/statistical style results
    if isinstance(data, dict) and ("profit_factor" in data or "return_pct" in data):
        profit_factor = _safe_float(data.get("profit_factor"), None)  # type: ignore[arg-type]
        return_pct = _safe_float(data.get("return_pct"), None)  # type: ignore[arg-type]
        max_drawdown_pct = _safe_float(data.get("max_drawdown"), None)  # negative values expected
        win_rate_pct = _safe_float(data.get("win_rate"), None)
        buy_hold_return_pct = _safe_float(data.get("buy_hold_return"), None)
        total_trades = int(data.get("total_trades")) if isinstance(data.get("total_trades"), (int, float)) else None
        exposure_time_pct = _safe_float(data.get("exposure_time"), None)

    # Walkforward performance results
    perf = data.get("performance_summary") if isinstance(data, dict) else None
    if isinstance(perf, dict):
        wf_avg_return_pct = _safe_float(perf.get("avg_return"), None)
        wf_avg_profit_factor = _safe_float(perf.get("avg_profit_factor"), None)
        total_windows = perf.get("total_windows") or 0
        positive_windows = perf.get("positive_windows") or 0
        try:
            wf_positive_window_ratio = float(positive_windows) / float(total_windows) if total_windows else None
        except Exception:
            wf_positive_window_ratio = None
        wf_performance_consistency = _safe_float(perf.get("performance_consistency"), None)

        # If return_pct not present from risk managed output, use walkforward avg_return as proxy
        if return_pct is None and wf_avg_return_pct is not None:
            return_pct = wf_avg_return_pct
        # If profit_factor not present, use walkforward avg PF
        if profit_factor is None and wf_avg_profit_factor is not None:
            profit_factor = wf_avg_profit_factor

    return NormalizedEvaluation(
        strategy=strategy,
        engine=engine,
        data_file=data_file,
        symbol=symbol,
        timeframe=timeframe,
        profit_factor=profit_factor,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        buy_hold_return_pct=buy_hold_return_pct,
        total_trades=total_trades,
        exposure_time_pct=exposure_time_pct,
        wf_avg_return_pct=wf_avg_return_pct,
        wf_avg_profit_factor=wf_avg_profit_factor,
        wf_positive_window_ratio=wf_positive_window_ratio,
        wf_performance_consistency=wf_performance_consistency,
    )


def parse_result_json(path: str) -> List[NormalizedEvaluation]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    # Identify engine by directory name under results
    parts = path.replace("\\", "/").split("/")
    engine_dir = None
    for i, p in enumerate(parts):
        if p == "results" and i + 1 < len(parts):
            engine_dir = parts[i + 1]
            break
    engine = engine_dir or "unknown_engine"

    evals: List[NormalizedEvaluation] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                ev = _parse_single_result(item, path, engine)
                if ev is not None:
                    evals.append(ev)
    elif isinstance(data, dict):
        ev = _parse_single_result(data, path, engine)
        if ev is not None:
            evals.append(ev)
    return evals


@dataclass
class StrategyAggregate:
    strategy: str
    total_evaluations: int

    # Aggregated metrics (medians preferred for robustness)
    median_profit_factor: float
    median_return_pct: float
    median_max_drawdown_pct: float
    median_win_rate_pct: float
    percent_profitable_evaluations: float  # 0..1
    percent_outperformed_buyhold: float    # 0..1

    # Walkforward consistency
    median_wf_positive_window_ratio: float
    median_wf_consistency: float

    # Composite score (no Sharpe)
    score: float

    # Optional diagnostics
    engines_covered: List[str]


def score_strategy(agg: StrategyAggregate) -> float:
    score = 0.0

    # Profit Factor weighting
    pf = agg.median_profit_factor
    if pf >= 1.8:
        score += 2.5
    elif pf >= 1.5:
        score += 2.0
    elif pf >= 1.3:
        score += 1.0

    # Return strength (median)
    r = agg.median_return_pct
    if r >= 20:
        score += 1.5
    elif r >= 10:
        score += 1.0
    elif r >= 5:
        score += 0.5

    # Drawdown control (lower is better; values may be negative)
    dd = abs(agg.median_max_drawdown_pct)
    if dd <= 15:
        score += 1.0
    elif dd <= 25:
        score += 0.5

    # Win rate (not primary, but stabilizer)
    wr = agg.median_win_rate_pct
    if wr >= 55:
        score += 0.5
    elif wr >= 50:
        score += 0.25

    # Consistency across datasets
    if agg.percent_profitable_evaluations >= 0.65:
        score += 1.0
    elif agg.percent_profitable_evaluations >= 0.55:
        score += 0.5

    # Walkforward consistency (if available)
    if agg.median_wf_positive_window_ratio >= 0.6:
        score += 1.0
    elif agg.median_wf_positive_window_ratio >= 0.5:
        score += 0.5

    if agg.median_wf_consistency >= 0.0:  # prefer non-negative stability
        score += 0.25

    # Outperform buy & hold across datasets
    if agg.percent_outperformed_buyhold >= 0.6:
        score += 0.5
    elif agg.percent_outperformed_buyhold >= 0.5:
        score += 0.25

    return score


def aggregate_by_strategy(evals: List[NormalizedEvaluation]) -> List[StrategyAggregate]:
    by_strategy: Dict[str, List[NormalizedEvaluation]] = {}
    for e in evals:
        by_strategy.setdefault(e.strategy, []).append(e)

    aggregates: List[StrategyAggregate] = []
    for strategy, es in by_strategy.items():
        pfs = [e.profit_factor for e in es if e.profit_factor is not None]
        rets = [e.return_pct for e in es if e.return_pct is not None]
        dds = [e.max_drawdown_pct for e in es if e.max_drawdown_pct is not None]
        wrs = [e.win_rate_pct for e in es if e.win_rate_pct is not None]
        wf_pos = [e.wf_positive_window_ratio for e in es if e.wf_positive_window_ratio is not None]
        wf_cons = [e.wf_performance_consistency for e in es if e.wf_performance_consistency is not None]

        # Profitability/outperformance rates
        profitable = 0
        outperformed_bh = 0
        total = 0
        engines: set[str] = set()
        for e in es:
            total += 1
            engines.add(e.engine)
            if e.return_pct is not None and e.return_pct > 0:
                profitable += 1
            if (e.return_pct is not None and e.buy_hold_return_pct is not None
                and e.return_pct > e.buy_hold_return_pct):
                outperformed_bh += 1

        agg = StrategyAggregate(
            strategy=strategy,
            total_evaluations=total,
            median_profit_factor=_median([_safe_float(v) for v in pfs], default=0.0),
            median_return_pct=_median([_safe_float(v) for v in rets], default=0.0),
            median_max_drawdown_pct=_median([_safe_float(v) for v in dds], default=0.0),
            median_win_rate_pct=_median([_safe_float(v) for v in wrs], default=0.0),
            percent_profitable_evaluations=(profitable / total) if total else 0.0,
            percent_outperformed_buyhold=(outperformed_bh / total) if total else 0.0,
            median_wf_positive_window_ratio=_median([_safe_float(v) for v in wf_pos], default=0.0),
            median_wf_consistency=_median([_safe_float(v) for v in wf_cons], default=-1.0),
            score=0.0,
            engines_covered=sorted(list(engines)),
        )
        agg.score = score_strategy(agg)
        aggregates.append(agg)

    return aggregates


def rank_strategies(aggregates: List[StrategyAggregate]) -> List[StrategyAggregate]:
    return sorted(
        aggregates,
        key=lambda a: (
            -a.score,
            -a.median_profit_factor,
            -a.median_return_pct,
            a.median_max_drawdown_pct,
        ),
    )


def write_reports(aggregates: List[StrategyAggregate], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"winners_summary_{ts}.json")
    md_path = os.path.join(output_dir, f"winners_summary_{ts}.md")

    # JSON report
    payload = {
        "generated_at": datetime.now().isoformat(),
        "total_strategies": len(aggregates),
        "strategies": [
            {
                **asdict(a),
                # Cast ratios to percentages for readability in JSON too
                "percent_profitable_evaluations": round(100 * a.percent_profitable_evaluations, 2),
                "percent_outperformed_buyhold": round(100 * a.percent_outperformed_buyhold, 2),
            }
            for a in aggregates
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown report (top 25 by default)
    lines: List[str] = []
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("Top Strategies (ranked, excluding Sharpe):")
    lines.append("")
    header = (
        "| Rank | Strategy | Score | PF (med) | Return% (med) | MaxDD% (med) | Win% (med) | Profitable% | WF+Win% (med) | Engines |"
    )
    sep = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)
    for idx, a in enumerate(aggregates[:25], start=1):
        lines.append(
            "| {rank} | {name} | {score:.2f} | {pf:.2f} | {ret:.2f} | {dd:.2f} | {wr:.2f} | {prof:.1f}% | {wfpos:.1f}% | {engs} |".format(
                rank=idx,
                name=a.strategy,
                score=a.score,
                pf=a.median_profit_factor,
                ret=a.median_return_pct,
                dd=a.median_max_drawdown_pct,
                wr=a.median_win_rate_pct,
                prof=100 * a.percent_profitable_evaluations,
                wfpos=100 * a.median_wf_positive_window_ratio,
                engs=",".join(a.engines_covered),
            )
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main() -> None:
    print(f"Scanning results under: {RESULTS_ROOT}")
    files = discover_json_files(RESULTS_ROOT)
    print(f"Found {len(files)} result files")

    evaluations: List[NormalizedEvaluation] = []
    for fp in files:
        evs = parse_result_json(fp)
        if evs:
            evaluations.extend(evs)

    if not evaluations:
        print("No evaluable results found.")
        return

    aggregates = aggregate_by_strategy(evaluations)
    ranked = rank_strategies(aggregates)

    output_dir = os.path.join(RESULTS_ROOT, "winners")
    json_path, md_path = write_reports(ranked, output_dir)

    print("\nWinner selection complete.")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()


