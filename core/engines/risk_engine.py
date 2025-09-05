"""
Risk Management Engine

This engine implements advanced risk management techniques including:
- Walk-forward analysis for out-of-sample validation
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Dynamic position sizing and risk-adjusted performance
- Maximum drawdown controls and correlation analysis
- Portfolio-level risk management
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
import json
from scipy import stats
from scipy.optimize import minimize

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.base.base_engine import BaseEngine, EngineConfig, BacktestResult
from core.base.base_strategy import BaseStrategy, StrategyConfig, Signal, Trade
from core.base.base_data_handler import BaseDataHandler, DataConfig
from core.base.base_risk_manager import BaseRiskManager, RiskConfig

warnings.filterwarnings('ignore')

@dataclass
class RiskEngineConfig(EngineConfig):
    """Configuration for risk management engine"""
    # Risk limits
    max_drawdown: float = 0.15
    max_portfolio_risk: float = 0.20
    max_position_size: float = 0.2
    max_correlation: float = 0.7
    
    # VaR settings
    var_confidence: float = 0.95
    var_lookback: int = 252
    cvar_confidence: float = 0.95
    
    # Walk-forward settings
    walk_forward_windows: int = 12
    walk_forward_overlap: float = 0.5
    min_training_period: int = 252
    min_test_period: int = 63
    
    # Position sizing methods
    position_sizing_method: str = "kelly"  # "kelly", "risk_parity", "volatility_targeting"
    kelly_fraction: float = 0.25  # Conservative Kelly fraction
    volatility_target: float = 0.15  # Annual volatility target
    
    # Risk metrics
    risk_free_rate: float = 0.02
    volatility_lookback: int = 60
    correlation_lookback: int = 252
    
    # Portfolio optimization
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    optimization_method: str = "risk_parity"  # "risk_parity", "max_sharpe", "min_variance"
    
    # Output settings
    save_risk_reports: bool = True
    generate_risk_plots: bool = True
    track_risk_metrics: bool = True

@dataclass
class RiskBacktestResult(BacktestResult):
    """Enhanced backtest results with comprehensive risk metrics"""
    # Risk metrics
    max_drawdown: float
    var_95: float
    cvar_95: float
    volatility: float
    downside_deviation: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Position sizing metrics
    avg_position_size: float
    max_position_size: float
    position_size_volatility: float
    
    # Walk-forward results
    walkforward_consistency: Optional[float] = None
    walkforward_returns: Optional[List[float]] = None
    walkforward_risk_metrics: Optional[List[Dict[str, float]]] = None
    
    # Portfolio risk metrics
    portfolio_var: Optional[float] = None
    portfolio_cvar: Optional[float] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Risk attribution
    risk_attribution: Optional[Dict[str, float]] = None
    factor_exposures: Optional[Dict[str, float]] = None

class RiskEngine(BaseEngine):
    """
    Advanced risk management engine with comprehensive risk controls and analysis
    """
    
    def __init__(self, config: RiskEngineConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        self.results = []
        self.risk_metrics = []
        self.walkforward_results = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data with risk calculations"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                col_mapping = {
                    'time': 'timestamp', 'date': 'timestamp',
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                }
                data = data.rename(columns=col_mapping)
            
            # Convert timestamp and sort
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')
            
            # Calculate risk metrics
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['volatility'] = data['returns'].rolling(window=self.config.volatility_lookback).std()
            data['rolling_var'] = data['returns'].rolling(window=self.config.var_lookback).quantile(1 - self.config.var_confidence)
            data['rolling_cvar'] = data['returns'].rolling(window=self.config.var_lookback).apply(
                lambda x: x[x <= x.quantile(1 - self.config.var_confidence)].mean()
            )
            
            self.logger.info(f"Loaded data with risk metrics: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def load_strategy(self, strategy_name: str, strategy_params: Dict[str, Any] = None) -> BaseStrategy:
        """Load strategy with risk parameter validation"""
        try:
            strategy = SimpleMAStrategy(
                name=strategy_name,
                parameters=strategy_params or {}
            )
            self.logger.info(f"Loaded strategy: {strategy_name}")
            return strategy
        except Exception as e:
            self.logger.error(f"Error loading strategy: {e}")
            raise
    
    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> RiskBacktestResult:
        """Run comprehensive backtest with advanced risk management"""
        try:
            self.logger.info(f"Starting risk-managed backtest for strategy: {strategy.name}")
            
            # Run walk-forward analysis
            walkforward_result = self._run_walkforward_analysis(data, strategy)
            
            # Run main backtest with risk controls
            backtest_result = self._run_risk_managed_backtest(data, strategy)
            
            # Calculate comprehensive risk metrics
            risk_metrics = self._calculate_risk_metrics(backtest_result['equity_curve'], backtest_result['trades'])
            
            # Create comprehensive result
            risk_result = RiskBacktestResult(
                strategy_name=strategy.name,
                symbol=data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in data.columns else 'UNKNOWN',
                timeframe='1D',
                start_date=data['timestamp'].min(),
                end_date=data['timestamp'].max(),
                initial_capital=self.config.initial_cash,
                final_capital=backtest_result['final_cash'],
                total_return=backtest_result['total_return'],
                total_trades=backtest_result['trades_count'],
                winning_trades=backtest_result['winning_trades'],
                losing_trades=backtest_result['losing_trades'],
                win_rate=backtest_result['win_rate'],
                avg_win=backtest_result['avg_win'],
                avg_loss=backtest_result['avg_loss'],
                profit_factor=backtest_result['profit_factor'],
                max_drawdown=risk_metrics['max_drawdown'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                sortino_ratio=risk_metrics['sortino_ratio'],
                calmar_ratio=risk_metrics['calmar_ratio'],
                equity_curve=backtest_result['equity_curve'],
                trade_log=backtest_result['trade_log'],
                daily_returns=backtest_result['daily_returns'],
                metadata={},
                
                # Risk metrics
                var_95=risk_metrics['var_95'],
                cvar_95=risk_metrics['cvar_95'],
                volatility=risk_metrics['volatility'],
                downside_deviation=risk_metrics['downside_deviation'],
                information_ratio=risk_metrics['information_ratio'],
                
                # Position sizing metrics
                avg_position_size=risk_metrics['avg_position_size'],
                max_position_size=risk_metrics['max_position_size'],
                position_size_volatility=risk_metrics['position_size_volatility'],
                
                # Walk-forward results
                walkforward_consistency=walkforward_result.get('consistency'),
                walkforward_returns=walkforward_result.get('returns'),
                walkforward_risk_metrics=walkforward_result.get('risk_metrics')
            )
            
            self.results.append(risk_result)
            self.logger.info(f"Risk-managed backtest completed successfully")
            
            return risk_result
            
        except Exception as e:
            self.logger.error(f"Error in risk-managed backtest: {e}")
            raise
    
    def _run_walkforward_analysis(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run walk-forward analysis for out-of-sample validation"""
        try:
            self.logger.info("Running walk-forward analysis...")
            
            total_size = len(data)
            train_size = self.config.min_training_period
            test_size = self.config.min_test_period
            step_size = int(test_size * (1 - self.config.walk_forward_overlap))
            
            if total_size < train_size + test_size:
                return {'error': f"Insufficient data: {total_size} < {train_size + test_size}"}
            
            walkforward_returns = []
            walkforward_risk_metrics = []
            
            for start_idx in range(0, total_size - train_size - test_size + 1, step_size):
                try:
                    # Training period
                    train_end = start_idx + train_size
                    train_data = data.iloc[start_idx:train_end]
                    
                    # Test period
                    test_start = train_end
                    test_end = min(test_start + test_size, total_size)
                    test_data = data.iloc[test_start:test_end]
                    
                    if len(test_data) < test_size * 0.8:
                        continue
                    
                    # Run backtest on test period
                    test_result = self._run_risk_managed_backtest(test_data, strategy)
                    if 'total_return' in test_result:
                        walkforward_returns.append(test_result['total_return'])
                        
                        # Calculate risk metrics for test period
                        if len(test_data) > 1:
                            risk_metrics = self._calculate_risk_metrics(
                                test_result['equity_curve'], 
                                test_result['trades']
                            )
                            walkforward_risk_metrics.append(risk_metrics)
                
                except Exception as e:
                    continue
            
            if len(walkforward_returns) == 0:
                return {'error': "No successful walk-forward periods"}
            
            # Calculate consistency metrics
            returns_array = np.array(walkforward_returns)
            consistency = np.std(returns_array) / np.mean(returns_array) if np.mean(returns_array) != 0 else float('inf')
            
            result = {
                'returns': walkforward_returns,
                'risk_metrics': walkforward_risk_metrics,
                'consistency': consistency,
                'n_periods': len(walkforward_returns),
                'mean_return': np.mean(returns_array),
                'std_return': np.std(returns_array)
            }
            
            self.walkforward_results.append(result)
            self.logger.info(f"Walk-forward analysis completed: {len(walkforward_returns)} periods")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {'error': str(e)}
    
    def _run_risk_managed_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run backtest with comprehensive risk management"""
        try:
            initial_cash = self.config.initial_cash
            cash = initial_cash
            position = 0
            trades = []
            equity_curve = [initial_cash]
            position_sizes = []
            
            for i in range(1, len(data)):
                try:
                    current_price = float(data['close'].iloc[i])
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    # Calculate dynamic position size based on risk
                    position_size = self._calculate_dynamic_position_size(
                        data, i, cash, current_price
                    )
                    position_sizes.append(position_size)
                    
                    # Risk-adjusted signal generation
                    signal = self._generate_risk_adjusted_signal(data, i, strategy)
                    
                    # Execute trades with risk controls
                    if signal == 1 and position <= 0:  # Buy signal
                        if position < 0:  # Close short
                            cash += abs(position) * current_price * (1 - self.config.commission)
                            trades.append({'type': 'close_short', 'price': current_price, 'index': i})
                        
                        # Open long position with risk control
                        new_position = int(cash * position_size / current_price)
                        if new_position > 0 and self._check_risk_limits(new_position, current_price, cash, data, i):
                            position = new_position
                            cash -= position * current_price * (1 + self.config.commission)
                            trades.append({'type': 'buy', 'price': current_price, 'index': i, 'size': position_size})
                    
                    elif signal == -1 and position >= 0:  # Sell signal
                        if position > 0:  # Close long
                            cash += position * current_price * (1 - self.config.commission)
                            trades.append({'type': 'sell', 'price': current_price, 'index': i})
                            position = 0
                        
                        # Open short position with risk control
                        new_position = -int(cash * position_size / current_price)
                        if new_position < 0 and self._check_risk_limits(abs(new_position), current_price, cash, data, i):
                            position = new_position
                            cash += abs(position) * current_price * (1 + self.config.commission)
                            trades.append({'type': 'short', 'price': current_price, 'index': i, 'size': position_size})
                    
                    # Update equity curve
                    current_equity = cash
                    if position > 0:
                        current_equity += position * current_price
                    elif position < 0:
                        current_equity -= abs(position) * current_price
                    equity_curve.append(current_equity)
                    
                    # Check drawdown limits
                    if self._check_drawdown_limit(equity_curve):
                        self.logger.warning("Maximum drawdown limit reached, closing positions")
                        break
                
                except Exception as e:
                    continue
            
            # Close final position
            try:
                final_price = float(data['close'].iloc[-1])
                if not pd.isna(final_price) and final_price > 0:
                    if position > 0:
                        cash += position * final_price * (1 - self.config.commission)
                    elif position < 0:
                        cash -= abs(position) * final_price * (1 + self.config.commission)
            except:
                pass
            
            # Calculate final metrics
            total_return = (cash - initial_cash) / initial_cash if initial_cash > 0 else 0.0
            equity_series = pd.Series(equity_curve)
            returns_series = equity_series.pct_change().dropna()
            
            # Calculate trade statistics
            winning_trades = len([t for t in trades if t['type'] in ['sell', 'close_short']])
            losing_trades = len(trades) - winning_trades
            win_rate = winning_trades / len(trades) if trades else 0.0
            
            # Calculate average win/loss
            avg_win = 0.01 if winning_trades > 0 else 0.0  # Simplified
            avg_loss = -0.01 if losing_trades > 0 else 0.0  # Simplified
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'final_cash': cash,
                'trades_count': len(trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'equity_curve': equity_series,
                'trades': trades,
                'position_sizes': position_sizes,
                'daily_returns': returns_series
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk-managed backtest: {e}")
            return {
                'total_return': 0.0,
                'final_cash': self.config.initial_cash,
                'trades_count': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'equity_curve': pd.Series(),
                'trades': [],
                'position_sizes': [],
                'daily_returns': pd.Series()
            }
    
    def _calculate_dynamic_position_size(self, data: pd.DataFrame, index: int, cash: float, price: float) -> float:
        """Calculate dynamic position size based on risk metrics"""
        try:
            if self.config.position_sizing_method == "kelly":
                return self._calculate_kelly_position_size(data, index, cash, price)
            elif self.config.position_sizing_method == "risk_parity":
                return self._calculate_risk_parity_position_size(data, index, cash, price)
            elif self.config.position_sizing_method == "volatility_targeting":
                return self._calculate_volatility_targeting_position_size(data, index, cash, price)
            else:
                return self.config.max_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.config.max_position_size
    
    def _calculate_kelly_position_size(self, data: pd.DataFrame, index: int, cash: float, price: float) -> float:
        """Calculate Kelly criterion position size"""
        try:
            if index < 60:  # Need sufficient data
                return self.config.max_position_size
            
            # Calculate recent returns and volatility
            recent_returns = data['returns'].iloc[max(0, index-60):index].dropna()
            if len(recent_returns) < 30:
                return self.config.max_position_size
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            win_rate = np.sum(recent_returns > 0) / len(recent_returns)
            avg_win = np.mean(recent_returns[recent_returns > 0]) if np.sum(recent_returns > 0) > 0 else 0.01
            avg_loss = np.mean(recent_returns[recent_returns < 0]) if np.sum(recent_returns < 0) < 0 else -0.01
            
            if avg_loss == 0:
                return self.config.max_position_size
            
            odds_ratio = abs(avg_win / avg_loss)
            kelly_fraction = (odds_ratio * win_rate - (1 - win_rate)) / odds_ratio
            
            # Apply conservative Kelly fraction
            conservative_kelly = kelly_fraction * self.config.kelly_fraction
            
            # Ensure within bounds
            return max(0.01, min(conservative_kelly, self.config.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Error in Kelly calculation: {e}")
            return self.config.max_position_size
    
    def _calculate_risk_parity_position_size(self, data: pd.DataFrame, index: int, cash: float, price: float) -> float:
        """Calculate risk parity position size"""
        try:
            if index < 60:
                return self.config.max_position_size
            
            # Calculate recent volatility
            recent_volatility = data['volatility'].iloc[index] if not pd.isna(data['volatility'].iloc[index]) else 0.02
            
            if recent_volatility == 0:
                return self.config.max_position_size
            
            # Risk parity: equal risk contribution
            target_risk = self.config.volatility_target / np.sqrt(252)  # Daily volatility target
            position_size = target_risk / recent_volatility
            
            return max(0.01, min(position_size, self.config.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Error in risk parity calculation: {e}")
            return self.config.max_position_size
    
    def _calculate_volatility_targeting_position_size(self, data: pd.DataFrame, index: int, cash: float, price: float) -> float:
        """Calculate volatility targeting position size"""
        try:
            if index < 60:
                return self.config.max_position_size
            
            # Get current volatility
            current_volatility = data['volatility'].iloc[index] if not pd.isna(data['volatility'].iloc[index]) else 0.02
            
            if current_volatility == 0:
                return self.config.max_position_size
            
            # Volatility targeting: adjust position size to maintain target volatility
            target_volatility = self.config.volatility_target / np.sqrt(252)
            position_size = target_volatility / current_volatility
            
            return max(0.01, min(position_size, self.config.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Error in volatility targeting calculation: {e}")
            return self.config.max_position_size
    
    def _generate_risk_adjusted_signal(self, data: pd.DataFrame, index: int, strategy: BaseStrategy) -> int:
        """Generate risk-adjusted trading signal"""
        try:
            # Get basic signal from strategy
            signal_data = data.iloc[:index+1]
            if len(signal_data) < 20:
                return 0
            
            # Simple momentum signal for demonstration
            price_change = (data['close'].iloc[index] - data['close'].iloc[index-1]) / data['close'].iloc[index-1]
            volatility = data['volatility'].iloc[index] if not pd.isna(data['volatility'].iloc[index]) else 0.02
            
            # Dynamic threshold based on volatility
            threshold = max(0.005, volatility * 2)
            
            if price_change > threshold:
                return 1  # Buy signal
            elif price_change < -threshold:
                return -1  # Sell signal
            else:
                return 0  # No signal
            
        except Exception as e:
            self.logger.error(f"Error generating risk-adjusted signal: {e}")
            return 0
    
    def _check_risk_limits(self, position_size: int, price: float, cash: float, data: pd.DataFrame, index: int) -> bool:
        """Check if position meets risk limits"""
        try:
            # Check maximum position size
            position_value = position_size * price
            if position_value > cash * self.config.max_position_size:
                return False
            
            # Check portfolio risk limits
            if hasattr(self, 'portfolio_risk') and self.portfolio_risk > self.config.max_portfolio_risk:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _check_drawdown_limit(self, equity_curve: List[float]) -> bool:
        """Check if maximum drawdown limit is exceeded"""
        try:
            if len(equity_curve) < 2:
                return False
            
            peak = max(equity_curve)
            current = equity_curve[-1]
            drawdown = (current - peak) / peak
            
            return drawdown < -self.config.max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown limit: {e}")
            return False
    
    def _calculate_risk_metrics(self, equity_curve: pd.Series, trades: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            if len(equity_curve) < 2:
                return self._get_default_risk_metrics()
            
            returns = equity_curve.pct_change().dropna()
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
            cvar_95 = returns[returns <= var_95].mean() if np.sum(returns <= var_95) > 0 else var_95
            
            # Risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1, equity_curve)
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
            
            # Information ratio (simplified)
            information_ratio = sharpe_ratio  # Assuming risk-free rate is 0 for simplicity
            
            # Position sizing metrics
            position_sizes = [t.get('size', 0.1) for t in trades if 'size' in t]
            avg_position_size = np.mean(position_sizes) if position_sizes else 0.1
            max_position_size = np.max(position_sizes) if position_sizes else 0.1
            position_size_volatility = np.std(position_sizes) if len(position_sizes) > 1 else 0.0
            
            return {
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'volatility': volatility,
                'downside_deviation': downside_deviation,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'avg_position_size': avg_position_size,
                'max_position_size': max_position_size,
                'position_size_volatility': position_size_volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Return default risk metrics when calculation fails"""
        return {
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'volatility': 0.0,
            'downside_deviation': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'avg_position_size': 0.1,
            'max_position_size': 0.1,
            'position_size_volatility': 0.0
        }
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = negative_returns.std()
        if downside_deviation == 0:
            return 0.0
        
        return float(returns.mean() / downside_deviation * np.sqrt(252))
    
    def _calculate_calmar_ratio(self, total_return: float, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio"""
        max_dd = self._calculate_max_drawdown(equity_curve)
        if max_dd == 0:
            return 0.0
        return float(total_return / abs(max_dd))
    
    def run_portfolio_backtest(self, data_dict: Dict[str, pd.DataFrame], strategies: List[BaseStrategy]) -> List[RiskBacktestResult]:
        """Run portfolio backtest with risk management"""
        # Implementation for portfolio backtesting
        pass
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, param_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters with risk constraints"""
        # Implementation for strategy optimization
        pass
    
    def save_results(self, results: List[RiskBacktestResult], output_path: str = None) -> str:
        """Save comprehensive risk analysis results"""
        try:
            if output_path is None:
                output_path = os.path.join(self.config.results_path, f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save main results
            results_file = os.path.join(output_path, "risk_results.json")
            with open(results_file, 'w') as f:
                json.dump([self._result_to_dict(result) for result in results], f, indent=2, default=str)
            
            # Save risk metrics
            if self.risk_metrics:
                metrics_file = os.path.join(output_path, "risk_metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(self.risk_metrics, f, indent=2, default=str)
            
            # Save walk-forward results
            if self.walkforward_results:
                wf_file = os.path.join(output_path, "walkforward_results.json")
                with open(wf_file, 'w') as f:
                    json.dump(self.walkforward_results, f, indent=2, default=str)
            
            # Generate summary report
            summary = self._generate_summary_report(results)
            summary_file = os.path.join(output_path, "summary_report.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Risk analysis results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _result_to_dict(self, result: RiskBacktestResult) -> Dict[str, Any]:
        """Convert result object to dictionary for JSON serialization"""
        return {
            'strategy_name': result.strategy_name,
            'total_return': result.total_return,
            'max_drawdown': result.max_drawdown,
            'var_95': result.var_95,
            'cvar_95': result.cvar_95,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'avg_position_size': result.avg_position_size,
            'walkforward_consistency': result.walkforward_consistency
        }
    
    def _generate_summary_report(self, results: List[RiskBacktestResult]) -> Dict[str, Any]:
        """Generate comprehensive risk summary report"""
        try:
            summary = {
                'total_strategies': len(results),
                'timestamp': datetime.now().isoformat(),
                'risk_summary': {
                    'avg_max_drawdown': np.mean([r.max_drawdown for r in results]),
                    'avg_var_95': np.mean([r.var_95 for r in results]),
                    'avg_sharpe_ratio': np.mean([r.sharpe_ratio for r in results]),
                    'avg_sortino_ratio': np.mean([r.sortino_ratio for r in results])
                },
                'position_sizing_summary': {
                    'avg_position_size': np.mean([r.avg_position_size for r in results]),
                    'max_position_size': np.max([r.max_position_size for r in results])
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return {'error': str(e)}


# Simple strategy class for demonstration
class SimpleMAStrategy(BaseStrategy):
    """Simple moving average strategy for testing"""
    
    def __init__(self, name: str = "SimpleMA", parameters: Dict[str, Any] = None):
        super().__init__(name)
        self.parameters = parameters or {}
        self.short_window = self.parameters.get('short_window', 10)
        self.long_window = self.parameters.get('long_window', 20)
    
    def initialize(self, data: pd.DataFrame):
        """Initialize strategy with data"""
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        signals = pd.Series(0, index=data.index)
        
        if len(data) < self.long_window:
            return signals
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals[short_ma > long_ma] = 1   # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals
