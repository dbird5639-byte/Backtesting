import numpy as np
from typing import Dict, Any


def calculate_risk_metrics(stats: Dict[str, Any], *, max_drawdown_pct: float, position_size_pct: float, enable_position_sizing: bool) -> Dict[str, Any]:
    """Compute a consistent set of risk metrics from backtest stats.

    Parameters:
        stats: Backtest statistics dict
        max_drawdown_pct: Configured max portfolio drawdown threshold (e.g., 0.2 for 20%)
        position_size_pct: Position size fraction of equity (e.g., 0.02 for 2%)
        enable_position_sizing: Whether position sizing is active
    """
    risk_metrics: Dict[str, Any] = {}

    # Risk-adjusted returns
    volatility = stats.get('Volatility (Ann.) [%]', 0)
    if volatility and float(volatility) != 0:
        risk_metrics['risk_adjusted_return'] = float(stats.get('Return [%]', 0)) / float(volatility)

    # Max consecutive losses analysis (approximate expectation)
    total_trades = int(stats.get('# Trades', 0) or 0)
    win_rate_pct = float(stats.get('Win Rate [%]', 0) or 0)
    win_rate = win_rate_pct / 100.0
    if total_trades > 0 and 0 < win_rate < 1:
        try:
            expected_max_consec_losses = int(np.log(max(total_trades, 1)) / np.log(1 / (1 - win_rate)))
        except Exception:
            expected_max_consec_losses = 0
        risk_metrics['expected_max_consecutive_losses'] = expected_max_consec_losses
        max_consec_losses = int(stats.get('Max. Consecutive Losses', 0) or 0)
        risk_metrics['consecutive_loss_ratio'] = (
            (max_consec_losses / expected_max_consec_losses) if expected_max_consec_losses > 0 else 0.0
        )

    # Drawdown risk score (relative to configured cap)
    max_dd = abs(float(stats.get('Max. Drawdown [%]', 0) or 0))
    risk_metrics['drawdown_risk_score'] = (max_dd / (max_drawdown_pct * 100.0)) if max_drawdown_pct else 0.0

    # Position sizing context
    if enable_position_sizing:
        ps = float(position_size_pct or 0)
        risk_metrics['position_size_analysis'] = {
            'position_size_pct': ps,
            'max_positions': int(1 / ps) if ps > 0 else 0,
            'diversification_score': min(1.0, (total_trades / 10.0))
        }

    # Aggregate risk management effectiveness score
    score = 0.0
    max_score = 5.0
    # Drawdown control
    if max_drawdown_pct:
        if max_dd < (max_drawdown_pct * 100.0):
            score += 1.0
        elif max_dd < (max_drawdown_pct * 150.0):
            score += 0.5
    # Consecutive loss control
    max_consec_losses = int(stats.get('Max. Consecutive Losses', 0) or 0)
    # Use a heuristic cap based on ps to avoid requiring full config here
    configured_cap = 5
    if max_consec_losses < configured_cap:
        score += 1.0
    elif max_consec_losses < configured_cap * 1.5:
        score += 0.5
    # Sharpe
    sharpe = float(stats.get('Sharpe Ratio', 0) or 0)
    if sharpe > 1.5:
        score += 1.0
    elif sharpe > 1.0:
        score += 0.5
    # Sortino
    sortino = float(stats.get('Sortino Ratio', 0) or 0)
    if sortino > 2.0:
        score += 1.0
    elif sortino > 1.5:
        score += 0.5
    # Calmar
    calmar = float(stats.get('Calmar Ratio', 0) or 0)
    if calmar > 1.0:
        score += 1.0
    elif calmar > 0.5:
        score += 0.5
    risk_metrics['risk_management_score'] = score / max_score

    return risk_metrics


