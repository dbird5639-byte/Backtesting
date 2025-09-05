#!/usr/bin/env python3
"""
Walk-Forward Analysis Utility

This module provides walk-forward analysis capabilities for backtesting strategies,
including out-of-sample testing and performance consistency analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardPeriod:
    """Walk-forward period data structure"""
    period_number: int
    training_start: int
    training_end: int
    testing_start: int
    testing_end: int
    training_data: pd.DataFrame
    testing_data: pd.DataFrame


@dataclass
class WalkForwardResult:
    """Walk-forward result structure"""
    strategy_name: str
    symbol: str
    timeframe: str
    
    # Training results
    training_periods: List[Dict[str, Any]]
    avg_training_return: float
    avg_training_sharpe: float
    avg_training_drawdown: float
    
    # Testing results
    testing_periods: List[Dict[str, Any]]
    avg_testing_return: float
    avg_testing_sharpe: float
    avg_testing_drawdown: float
    
    # Walk-forward metrics
    total_periods: int
    consistency_score: float  # How consistent performance is across periods
    degradation_score: float  # How much performance degrades from training to testing
    
    # Overall metrics
    overall_return: float
    overall_sharpe: float
    overall_drawdown: float
    overall_win_rate: float
    
    # Risk metrics
    max_consecutive_losses: int
    recovery_factor: float
    profit_factor: float


class WalkForwardAnalyzer:
    """Walk-forward analysis for trading strategies"""
    
    def __init__(self, training_ratio: float = 0.7, min_training_periods: int = 500,
                 min_testing_periods: int = 200, step_size: int = 100):
        """
        Initialize walk-forward analyzer.
        
        Args:
            training_ratio: Ratio of data to use for training (0.7 = 70% training, 30% testing)
            min_training_periods: Minimum number of periods required for training
            min_testing_periods: Minimum number of periods required for testing
            step_size: Number of periods to move forward in each iteration
        """
        self.training_ratio = training_ratio
        self.min_training_periods = min_training_periods
        self.min_testing_periods = min_testing_periods
        self.step_size = step_size
    
    def create_walk_forward_periods(self, data: pd.DataFrame) -> List[WalkForwardPeriod]:
        """
        Create walk-forward periods from data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of walk-forward periods
        """
        periods = []
        total_periods = len(data)
        
        if total_periods < self.min_training_periods + self.min_testing_periods:
            logger.warning(f"Insufficient data for walk-forward analysis. Need at least {self.min_training_periods + self.min_testing_periods} periods, got {total_periods}")
            return periods
        
        # Calculate initial training size
        training_size = int(total_periods * self.training_ratio)
        training_size = max(training_size, self.min_training_periods)
        
        period_number = 0
        training_start = 0
        
        while True:
            training_end = training_start + training_size
            testing_start = training_end
            testing_end = testing_start + self.min_testing_periods
            
            # Check if we have enough data for testing
            if testing_end > total_periods:
                break
            
            # Create period
            training_data = data.iloc[training_start:training_end].copy()
            testing_data = data.iloc[testing_start:testing_end].copy()
            
            period = WalkForwardPeriod(
                period_number=period_number,
                training_start=training_start,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=testing_end,
                training_data=training_data,
                testing_data=testing_data
            )
            
            periods.append(period)
            
            # Move to next period
            period_number += 1
            training_start += self.step_size
            
            # Check if we still have enough data for training
            if training_start + training_size > total_periods:
                break
        
        logger.info(f"Created {len(periods)} walk-forward periods")
        return periods
    
    def run_walk_forward_analysis(self, strategy_class: Callable, data: pd.DataFrame,
                                 strategy_params: Optional[Dict[str, Any]] = None,
                                 symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> WalkForwardResult:
        """
        Run walk-forward analysis on a strategy.
        
        Args:
            strategy_class: Strategy class to test
            data: Market data
            strategy_params: Strategy parameters
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Walk-forward result object
        """
        logger.info(f"Starting walk-forward analysis for {strategy_class.__name__}")
        
        # Create walk-forward periods
        periods = self.create_walk_forward_periods(data)
        
        if not periods:
            logger.error("No walk-forward periods created")
            return None
        
        # Initialize results storage
        training_results = []
        testing_results = []
        
        # Run analysis for each period
        for period in periods:
            logger.info(f"Processing period {period.period_number + 1}/{len(periods)}")
            
            try:
                # Train strategy on training data
                training_result = self._run_training_period(strategy_class, period.training_data, strategy_params)
                if training_result:
                    training_result['period_number'] = period.period_number
                    training_results.append(training_result)
                
                # Test strategy on testing data
                testing_result = self._run_testing_period(strategy_class, period.training_data, 
                                                       period.testing_data, strategy_params)
                if testing_result:
                    testing_result['period_number'] = period.period_number
                    testing_results.append(testing_result)
                
            except Exception as e:
                logger.error(f"Error processing period {period.period_number}: {e}")
                continue
        
        # Calculate walk-forward metrics
        result = self._calculate_walk_forward_metrics(
            strategy_class.__name__, symbol, timeframe,
            training_results, testing_results
        )
        
        logger.info(f"Walk-forward analysis completed for {strategy_class.__name__}")
        return result
    
    def _run_training_period(self, strategy_class: Callable, training_data: pd.DataFrame,
                            strategy_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run training period and return results."""
        try:
            # Create strategy instance
            strategy = strategy_class(strategy_params or {})
            
            # Run backtest on training data
            if hasattr(strategy, 'backtest'):
                result = strategy.backtest(training_data)
            else:
                # Fallback to basic signal calculation
                signals = strategy.calculate_signals(training_data)
                result = self._calculate_basic_metrics(signals, training_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in training period: {e}")
            return None
    
    def _run_testing_period(self, strategy_class: Callable, training_data: pd.DataFrame,
                           testing_data: pd.DataFrame, strategy_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run testing period and return results."""
        try:
            # Create strategy instance
            strategy = strategy_class(strategy_params or {})
            
            # Run backtest on testing data
            if hasattr(strategy, 'backtest'):
                result = strategy.backtest(testing_data)
            else:
                # Fallback to basic signal calculation
                signals = strategy.calculate_signals(testing_data)
                result = self._calculate_basic_metrics(signals, testing_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in testing period: {e}")
            return None
    
    def _calculate_basic_metrics(self, signals: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic performance metrics from signals."""
        if signals is None or len(signals) == 0:
            return {}
        
        # Simple P&L calculation
        returns = signals.shift(1) * data['close'].pct_change()
        cumulative_return = (1 + returns).prod() - 1
        
        # Calculate basic metrics
        result = {
            'total_return': cumulative_return,
            'final_cash': 100000 * (1 + cumulative_return),  # Assume $100k starting capital
            'trades_count': len(signals[signals != 0]),
            'equity_curve': (1 + returns).cumprod().values.tolist()
        }
        
        # Calculate additional metrics if possible
        if len(returns) > 1:
            result['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
            result['win_rate'] = len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        
        return result
    
    def _calculate_walk_forward_metrics(self, strategy_name: str, symbol: str, timeframe: str,
                                      training_results: List[Dict[str, Any]],
                                      testing_results: List[Dict[str, Any]]) -> WalkForwardResult:
        """Calculate comprehensive walk-forward metrics."""
        
        # Calculate training averages
        training_returns = [r.get('total_return', 0) for r in training_results if 'total_return' in r]
        training_sharpes = [r.get('sharpe_ratio', 0) for r in training_results if 'sharpe_ratio' in r]
        training_drawdowns = [r.get('max_drawdown', 0) for r in training_results if 'max_drawdown' in r]
        
        avg_training_return = np.mean(training_returns) if training_returns else 0
        avg_training_sharpe = np.mean(training_sharpes) if training_sharpes else 0
        avg_training_drawdown = np.mean(training_drawdowns) if training_drawdowns else 0
        
        # Calculate testing averages
        testing_returns = [r.get('total_return', 0) for r in testing_results if 'total_return' in r]
        testing_sharpes = [r.get('sharpe_ratio', 0) for r in testing_results if 'sharpe_ratio' in r]
        testing_drawdowns = [r.get('max_drawdown', 0) for r in testing_results if 'max_drawdown' in r]
        
        avg_testing_return = np.mean(testing_returns) if testing_returns else 0
        avg_testing_sharpe = np.mean(testing_sharpes) if testing_sharpes else 0
        avg_testing_drawdown = np.mean(testing_drawdowns) if testing_drawdowns else 0
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(testing_returns)
        
        # Calculate degradation score
        degradation_score = self._calculate_degradation_score(training_returns, testing_returns)
        
        # Calculate overall metrics
        all_returns = training_returns + testing_returns
        overall_return = np.mean(all_returns) if all_returns else 0
        overall_sharpe = np.mean(training_sharpes + testing_sharpes) if (training_sharpes + testing_sharpes) else 0
        overall_drawdown = np.mean(training_drawdowns + testing_drawdowns) if (training_drawdowns + testing_drawdowns) else 0
        
        # Calculate win rate
        all_trades = []
        for result in training_results + testing_results:
            if 'trades' in result:
                all_trades.extend(result['trades'])
        
        overall_win_rate = self._calculate_win_rate(all_trades)
        
        # Calculate risk metrics
        max_consecutive_losses = self._calculate_max_consecutive_losses(testing_returns)
        recovery_factor = self._calculate_recovery_factor(testing_returns)
        profit_factor = self._calculate_profit_factor(testing_returns)
        
        return WalkForwardResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            training_periods=training_results,
            avg_training_return=avg_training_return,
            avg_training_sharpe=avg_training_sharpe,
            avg_training_drawdown=avg_training_drawdown,
            testing_periods=testing_results,
            avg_testing_return=avg_testing_return,
            avg_testing_sharpe=avg_testing_sharpe,
            avg_testing_drawdown=avg_testing_drawdown,
            total_periods=len(training_results),
            consistency_score=consistency_score,
            degradation_score=degradation_score,
            overall_return=overall_return,
            overall_sharpe=overall_sharpe,
            overall_drawdown=overall_drawdown,
            overall_win_rate=overall_win_rate,
            max_consecutive_losses=max_consecutive_losses,
            recovery_factor=recovery_factor,
            profit_factor=profit_factor
        )
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score based on return stability."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        
        # Convert to 0-1 scale where 1 is most consistent
        consistency_score = max(0.0, 1.0 - min(cv, 1.0))
        
        return consistency_score
    
    def _calculate_degradation_score(self, training_returns: List[float], 
                                   testing_returns: List[float]) -> float:
        """Calculate performance degradation from training to testing."""
        if not training_returns or not testing_returns:
            return 0.0
        
        avg_training = np.mean(training_returns)
        avg_testing = np.mean(testing_returns)
        
        if avg_training == 0:
            return 0.0
        
        # Calculate degradation as percentage
        degradation = (avg_training - avg_testing) / abs(avg_training)
        
        # Normalize to 0-1 scale where 1 is no degradation
        degradation_score = max(0.0, 1.0 - degradation)
        
        return degradation_score
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(trades)
    
    def _calculate_max_consecutive_losses(self, returns: List[float]) -> int:
        """Calculate maximum consecutive losses."""
        if not returns:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_recovery_factor(self, returns: List[float]) -> float:
        """Calculate recovery factor."""
        if not returns:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + np.array(returns))
        max_drawdown = np.min(cumulative_returns)
        
        if max_drawdown == 0:
            return 0.0
        
        final_return = cumulative_returns[-1]
        recovery_factor = final_return / abs(max_drawdown)
        
        return recovery_factor
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor."""
        if not returns:
            return 0.0
        
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        total_profit = sum(positive_returns) if positive_returns else 0
        total_loss = abs(sum(negative_returns)) if negative_returns else 0
        
        if total_loss == 0:
            return 0.0
        
        profit_factor = total_profit / total_loss
        return profit_factor
    
    def generate_walk_forward_report(self, result: WalkForwardResult) -> str:
        """Generate a comprehensive walk-forward report."""
        if not result:
            return "No walk-forward results available"
        
        report = f"""
Walk-Forward Analysis Report
============================

Strategy: {result.strategy_name}
Symbol: {result.symbol}
Timeframe: {result.timeframe}
Total Periods: {result.total_periods}

Training Performance:
- Average Return: {result.avg_training_return:.2%}
- Average Sharpe Ratio: {result.avg_training_sharpe:.3f}
- Average Max Drawdown: {result.avg_training_drawdown:.2%}

Testing Performance:
- Average Return: {result.avg_testing_return:.2%}
- Average Sharpe Ratio: {result.avg_testing_sharpe:.3f}
- Average Max Drawdown: {result.avg_testing_drawdown:.2%}

Walk-Forward Metrics:
- Consistency Score: {result.consistency_score:.3f} (0-1, higher is better)
- Degradation Score: {result.degradation_score:.3f} (0-1, higher is better)

Overall Performance:
- Total Return: {result.overall_return:.2%}
- Sharpe Ratio: {result.overall_sharpe:.3f}
- Max Drawdown: {result.overall_drawdown:.2%}
- Win Rate: {result.overall_win_rate:.2%}

Risk Metrics:
- Max Consecutive Losses: {result.max_consecutive_losses}
- Recovery Factor: {result.recovery_factor:.3f}
- Profit Factor: {result.profit_factor:.3f}

Analysis:
"""
        
        # Add analysis based on metrics
        if result.consistency_score > 0.7:
            report += "- High consistency across periods\n"
        elif result.consistency_score > 0.4:
            report += "- Moderate consistency across periods\n"
        else:
            report += "- Low consistency across periods\n"
        
        if result.degradation_score > 0.8:
            report += "- Minimal performance degradation from training to testing\n"
        elif result.degradation_score > 0.6:
            report += "- Moderate performance degradation from training to testing\n"
        else:
            report += "- Significant performance degradation from training to testing\n"
        
        if result.overall_sharpe > 1.0:
            report += "- Good risk-adjusted returns\n"
        elif result.overall_sharpe > 0.5:
            report += "- Moderate risk-adjusted returns\n"
        else:
            report += "- Poor risk-adjusted returns\n"
        
        return report
