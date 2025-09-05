"""
Best token/timeframe pairs per strategy and recommended settings.

- Scans strategies directory to reflect the new setup (True_winners, Watchlist, etc.).
- Reads all JSON results and aggregates by (strategy, symbol, timeframe).
- Ranks the top pairs per strategy using a composite score (Sharpe excluded):
  Profit Factor, Return %, Drawdown %, Consistency, Walkforward positives.
- Extracts best available settings when present (e.g., risk_management_config) per pair.

Outputs:
- best_pairs_summary.json
- best_pairs_summary.md

Usage:
  python Backtesting/tool_box/analysis/best_pairs_report.py
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = r"C:\Users\andre\OneDrive\Desktop\Mastercode"
STRATEGIES_DIR = os.path.join(PROJECT_ROOT, "Backtesting", "strategies")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "Backtesting", "results")


def _is_finite_number(value: Any) -> bool:
    try:
        f = float(value)
        return math.isfinite(f)
    except Exception:
        return False


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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
    m = n // 2
    if n % 2:
        return float(finite[m])
    return float((finite[m - 1] + finite[m]) / 2.0)


def _mean(values: List[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if _is_finite_number(v)]
    if not finite:
        return default
    return float(sum(finite) / len(finite))


def list_strategies_layout(strategies_dir: str) -> Dict[str, List[str]]:
    layout: Dict[str, List[str]] = {}
    for sub in sorted(os.listdir(strategies_dir)):
        sub_path = os.path.join(strategies_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        py_files = [f for f in os.listdir(sub_path) if f.endswith('.py')]
        layout[sub] = sorted(py_files)
    return layout


def discover_json_files(results_root: str) -> List[str]:
    json_files: List[str] = []
    for root, dirs, files in os.walk(results_root):
        if os.path.basename(root) == "logs":
            continue
        for name in files:
            if not name.endswith('.json') or name.endswith('.tmp'):
                continue
            json_files.append(os.path.join(root, name))
    return json_files


@dataclass
class Eval:
    strategy: str
    engine: str
    symbol: Optional[str]
    timeframe: Optional[str]
    data_file: str
    profit_factor: Optional[float]
    return_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    win_rate_pct: Optional[float]
    buy_hold_return_pct: Optional[float]
    total_trades: Optional[int]
    exposure_time_pct: Optional[float]
    wf_positive_window_ratio: Optional[float]
    wf_consistency: Optional[float]
    risk_management_config: Optional[Dict[str, Any]]


def _engine_from_path(path: str) -> str:
    parts = path.replace('\\', '/').split('/')
    try:
        idx = parts.index('results')
        return parts[idx + 1]
    except Exception:
        return 'unknown_engine'


def _parse_single_record(d: Dict[str, Any], path: str) -> Optional[Eval]:
    strategy = d.get('strategy') or 'UNKNOWN'
    symbol = d.get('symbol')
    timeframe = d.get('timeframe')
    data_file = d.get('data_file') or os.path.splitext(os.path.basename(path))[0]
    engine = _engine_from_path(path)

    pf = _safe_float(d.get('profit_factor'))
    ret = _safe_float(d.get('return_pct'))
    dd = _safe_float(d.get('max_drawdown'))
    wr = _safe_float(d.get('win_rate'))
    bh = _safe_float(d.get('buy_hold_return'))
    tt = int(d['total_trades']) if isinstance(d.get('total_trades'), (int, float)) else None
    exp = _safe_float(d.get('exposure_time'))

    # walkforward
    wf_ratio = None
    wf_cons = None
    perf = d.get('performance_summary') if isinstance(d, dict) else None
    if isinstance(perf, dict):
        total_windows = perf.get('total_windows') or 0
        pos = perf.get('positive_windows') or 0
        try:
            wf_ratio = float(pos) / float(total_windows) if total_windows else None
        except Exception:
            wf_ratio = None
        wf_cons = _safe_float(perf.get('performance_consistency'))
        # Use walkforward avg return/pf if basic fields missing
        if ret is None:
            ret = _safe_float(perf.get('avg_return'))
        if pf is None:
            pf = _safe_float(perf.get('avg_profit_factor'))

    rmc = d.get('risk_management_config') if isinstance(d, dict) else None

    return Eval(
        strategy=strategy,
        engine=engine,
        symbol=symbol,
        timeframe=timeframe,
        data_file=data_file,
        profit_factor=pf,
        return_pct=ret,
        max_drawdown_pct=dd,
        win_rate_pct=wr,
        buy_hold_return_pct=bh,
        total_trades=tt,
        exposure_time_pct=exp,
        wf_positive_window_ratio=wf_ratio,
        wf_consistency=wf_cons,
        risk_management_config=rmc,
    )


def parse_result_file(path: str) -> List[Eval]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        return []
    evals: List[Eval] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                ev = _parse_single_record(item, path)
                if ev:
                    evals.append(ev)
    elif isinstance(data, dict):
        ev = _parse_single_record(data, path)
        if ev:
            evals.append(ev)
    return evals


@dataclass
class PairAggregate:
    strategy: str
    symbol: str
    timeframe: str
    total_evals: int
    median_profit_factor: float
    median_return_pct: float
    median_max_drawdown_pct: float
    median_win_rate_pct: float
    percent_profitable: float
    percent_outperform_bh: float
    median_wf_positive_ratio: float
    median_wf_consistency: float
    engines_covered: List[str]
    representative_settings: Optional[Dict[str, Any]]  # from best individual eval if available
    score: float


def score_pair(p: PairAggregate) -> float:
    score = 0.0
    # Profit Factor
    if p.median_profit_factor >= 1.8:
        score += 2.5
    elif p.median_profit_factor >= 1.5:
        score += 2.0
    elif p.median_profit_factor >= 1.3:
        score += 1.0
    # Return
    if p.median_return_pct >= 20:
        score += 1.5
    elif p.median_return_pct >= 10:
        score += 1.0
    elif p.median_return_pct >= 5:
        score += 0.5
    # Drawdown (lower better; values likely negative)
    dd = abs(p.median_max_drawdown_pct)
    if dd <= 15:
        score += 1.0
    elif dd <= 25:
        score += 0.5
    # Consistency and WF
    if p.percent_profitable >= 0.6:
        score += 0.75
    elif p.percent_profitable >= 0.5:
        score += 0.5
    if p.median_wf_positive_ratio >= 0.6:
        score += 0.75
    elif p.median_wf_positive_ratio >= 0.5:
        score += 0.5
    if p.median_wf_consistency >= 0:
        score += 0.25
    # Outperform BH
    if p.percent_outperform_bh >= 0.6:
        score += 0.25
    elif p.percent_outperform_bh >= 0.5:
        score += 0.1
    return score


def aggregate_pairs(evals: List[Eval]) -> List[PairAggregate]:
    # Group by (strategy, symbol, timeframe)
    groups: Dict[Tuple[str, str, str], List[Eval]] = {}
    for e in evals:
        if not e.strategy or not e.symbol or not e.timeframe:
            continue
        key = (e.strategy, e.symbol, e.timeframe)
        groups.setdefault(key, []).append(e)

    aggregates: List[PairAggregate] = []
    for (strategy, symbol, timeframe), es in groups.items():
        pfs = [e.profit_factor for e in es if e.profit_factor is not None]
        rets = [e.return_pct for e in es if e.return_pct is not None]
        dds = [e.max_drawdown_pct for e in es if e.max_drawdown_pct is not None]
        wrs = [e.win_rate_pct for e in es if e.win_rate_pct is not None]
        wf_pos = [e.wf_positive_window_ratio for e in es if e.wf_positive_window_ratio is not None]
        wf_cons = [e.wf_consistency for e in es if e.wf_consistency is not None]

        prof = 0
        out_bh = 0
        for e in es:
            if e.return_pct is not None and e.return_pct > 0:
                prof += 1
            if e.return_pct is not None and e.buy_hold_return_pct is not None and e.return_pct > e.buy_hold_return_pct:
                out_bh += 1
        total = len(es)
        engines = sorted(set(e.engine for e in es))

        # pick representative settings from best single eval by PF then Return (that has risk config)
        best_eval = None
        for cand in sorted(es, key=lambda x: (
            -(x.profit_factor or -1e9),
            -(x.return_pct or -1e9),
            (abs(x.max_drawdown_pct) if x.max_drawdown_pct is not None else 1e9)
        )):
            if cand.risk_management_config:
                best_eval = cand
                break

        agg = PairAggregate(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            total_evals=total,
            median_profit_factor=_median([float(v) for v in pfs], 0.0),
            median_return_pct=_median([float(v) for v in rets], 0.0),
            median_max_drawdown_pct=_median([float(v) for v in dds], 0.0),
            median_win_rate_pct=_median([float(v) for v in wrs], 0.0),
            percent_profitable=(prof / total) if total else 0.0,
            percent_outperform_bh=(out_bh / total) if total else 0.0,
            median_wf_positive_ratio=_median([float(v) for v in wf_pos], 0.0),
            median_wf_consistency=_median([float(v) for v in wf_cons], -1.0),
            engines_covered=engines,
            representative_settings=(best_eval.risk_management_config if best_eval else None),
            score=0.0,
        )
        agg.score = score_pair(agg)
        aggregates.append(agg)
    return aggregates


def build_reports(aggs: List[PairAggregate]) -> Tuple[Dict[str, Any], List[str]]:
    # Organize per strategy
    per_strategy: Dict[str, List[PairAggregate]] = {}
    for a in aggs:
        per_strategy.setdefault(a.strategy, []).append(a)
    # Rank pairs in each strategy
    for s in per_strategy:
        per_strategy[s] = sorted(
            per_strategy[s],
            key=lambda x: (-x.score, -x.median_profit_factor, -x.median_return_pct, abs(x.median_max_drawdown_pct)),
        )

    summary_json: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "strategies": {},
    }
    lines: List[str] = []
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("Best token/timeframe per strategy (Sharpe excluded):")
    lines.append("")
    lines.append("| Strategy | Pair | Score | PF(med) | Ret%(med) | MaxDD%(med) | Profitable% | WF+Win%(med) | Engines | Settings |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")

    for strategy, pairs in per_strategy.items():
        summary_json["strategies"][strategy] = {
            "top_pairs": [
                {
                    **asdict(p),
                    "percent_profitable": round(100 * p.percent_profitable, 2),
                    "percent_outperform_bh": round(100 * p.percent_outperform_bh, 2),
                }
                for p in pairs[:5]
            ]
        }
        for p in pairs[:5]:
            settings_preview = "-"
            if p.representative_settings:
                # compact display of core settings
                keys = [
                    "stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
                    "position_size_pct", "max_consecutive_losses", "max_drawdown_pct",
                ]
                parts = []
                for k in keys:
                    if k in p.representative_settings:
                        parts.append(f"{k}={p.representative_settings[k]}")
                settings_preview = ", ".join(parts) if parts else "custom"

            lines.append(
                "| {strategy} | {pair} | {score:.2f} | {pf:.2f} | {ret:.2f} | {dd:.2f} | {prof:.1f}% | {wf:.1f}% | {engs} | {settings} |".format(
                    strategy=strategy,
                    pair=f"{p.symbol}_{p.timeframe}",
                    score=p.score,
                    pf=p.median_profit_factor,
                    ret=p.median_return_pct,
                    dd=p.median_max_drawdown_pct,
                    prof=100 * p.percent_profitable,
                    wf=100 * p.median_wf_positive_ratio,
                    engs=",".join(p.engines_covered),
                    settings=settings_preview,
                )
            )

    return summary_json, lines


def main() -> None:
    # 1) Show strategies layout
    layout = list_strategies_layout(STRATEGIES_DIR)
    print("Strategies layout:")
    for group, files in layout.items():
        print(f"- {group}: {len(files)} files")

    # 2) Load results
    files = discover_json_files(RESULTS_ROOT)
    evals: List[Eval] = []
    for fp in files:
        evals.extend(parse_result_file(fp))
    print(f"Loaded {len(evals)} evaluations from {len(files)} files")

    if not evals:
        print("No evaluations found.")
        return

    # 3) Aggregate by (strategy, symbol, timeframe)
    aggs = aggregate_pairs(evals)
    print(f"Built {len(aggs)} pair aggregates")

    # 4) Build reports
    summary_json, lines = build_reports(aggs)
    out_dir = os.path.join(RESULTS_ROOT, "winners")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"best_pairs_summary_{ts}.json")
    md_path = os.path.join(out_dir, f"best_pairs_summary_{ts}.md")
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\nReports written:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()


