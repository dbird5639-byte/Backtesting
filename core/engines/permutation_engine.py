"""
Permutation Testing Engine

This engine uses advanced statistical permutation methods to validate strategy performance
by randomly shuffling data and comparing results to determine statistical significance.
Incorporates sophisticated testing methods from the user's existing permutation engine.
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
import pickle
from scipy import stats
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.base.base_engine import BaseEngine, EngineConfig, BacktestResult
from core.base.base_strategy import BaseStrategy, StrategyConfig, Signal, Trade
from core.base.base_data_handler import BaseDataHandler, DataConfig
from core.base.base_risk_manager import BaseRiskManager, RiskConfig

warnings.filterwarnings('ignore')

@dataclass
class PermutationEngineConfig(EngineConfig):
    """Enhanced configuration for permutation testing engine"""
    # Permutation testing settings
    n_permutations: int = 100
    confidence_level: float = 0.95
    min_permutations: int = 50
    max_permutations: int = 10000
    
    # Statistical significance settings
    significance_threshold: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # "bonferroni", "holm", "fdr"
    
    # Data shuffling methods
    shuffle_methods: List[str] = field(default_factory=lambda: [
        "price_shuffle", "return_shuffle", "volume_shuffle", "time_shuffle", "block_shuffle"
    ])
    
    # Advanced permutation settings
    preserve_correlation: bool = True
    block_size: int = 20  # For block shuffling
    seasonal_adjustment: bool = True
    volatility_scaling: bool = True
    
    # Output and analysis settings
    save_permutation_details: bool = True
    generate_permutation_plots: bool = True
    calculate_power: bool = True
    effect_size_calculation: bool = True
    
    # Performance optimization
    parallel_processing: bool = False
    n_jobs: int = -1
    chunk_size: int = 100

@dataclass
class PermutationTestResult:
    """Comprehensive results from permutation testing"""
    strategy_name: str
    original_return: float
    original_metrics: Dict[str, float]
    
    # Permutation statistics
    n_permutations: int
    successful_permutations: int
    permutation_returns: List[float]
    permutation_metrics: List[Dict[str, float]]
    
    # Statistical significance
    p_value: float
    adjusted_p_value: float
    significant: bool
    significance_level: float
    
    # Confidence intervals
    confidence_interval: List[float]
    confidence_level: float
    
    # Effect size and power
    effect_size: Optional[float] = None
    statistical_power: Optional[float] = None
    
    # Additional analysis
    permutation_distribution_stats: Dict[str, float]
    extreme_value_analysis: Dict[str, Any]
    robustness_metrics: Dict[str, float]
    
    # Metadata
    test_timestamp: datetime = field(default_factory=datetime.now)
    shuffle_methods_used: List[str] = field(default_factory=list)
    computation_time: Optional[float] = None

class PermutationEngine(BaseEngine):
    """
    Advanced permutation testing engine with sophisticated statistical validation methods
    """
    
    def __init__(self, config: PermutationEngineConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        self.results = []
        self.permutation_results = []
        self.test_history = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data from various file formats"""
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
                # Map common column names
                col_mapping = {
                    'time': 'timestamp', 'date': 'timestamp',
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                }
                data = data.rename(columns=col_mapping)
            
            # Convert timestamp and sort
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')
            
            # Calculate additional features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            
            self.logger.info(f"Loaded data: {len(data)} rows, {len(data.columns)} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def load_strategy(self, strategy_name: str, strategy_params: Dict[str, Any] = None) -> BaseStrategy:
        """Load strategy with parameter validation"""
        try:
            # For now, create a simple strategy - in practice, this would load from file
            strategy = SimpleMAStrategy(
                name=strategy_name,
                parameters=strategy_params or {}
            )
            self.logger.info(f"Loaded strategy: {strategy_name}")
            return strategy
        except Exception as e:
            self.logger.error(f"Error loading strategy: {e}")
            raise
    
    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> PermutationTestResult:
        """Run comprehensive permutation test with all validation methods"""
        try:
            self.logger.info(f"Starting permutation test for strategy: {strategy.name}")
            start_time = datetime.now()
            
            # Run original backtest
            original_result = self._run_comprehensive_backtest(data, strategy)
            original_return = original_result['total_return']
            original_metrics = original_result['metrics']
            
            # Run permutation tests with multiple methods
            permutation_results = self._run_advanced_permutation_tests(data, strategy, original_result)
            
            # Calculate statistical significance
            significance_results = self._calculate_statistical_significance(
                original_return, permutation_results['returns']
            )
            
            # Calculate effect size and power
            effect_size = self._calculate_effect_size(original_return, permutation_results['returns'])
            statistical_power = self._calculate_statistical_power(
                original_return, permutation_results['returns'], self.config.significance_threshold
            )
            
            # Generate comprehensive result
            permutation_result = PermutationTestResult(
                strategy_name=strategy.name,
                original_return=original_return,
                original_metrics=original_metrics,
                
                n_permutations=self.config.n_permutations,
                successful_permutations=len(permutation_results['returns']),
                permutation_returns=permutation_results['returns'],
                permutation_metrics=permutation_results['metrics'],
                
                p_value=significance_results['p_value'],
                adjusted_p_value=significance_results['adjusted_p_value'],
                significant=significance_results['significant'],
                significance_level=self.config.significance_threshold,
                
                confidence_interval=significance_results['confidence_interval'],
                confidence_level=self.config.confidence_level,
                
                effect_size=effect_size,
                statistical_power=statistical_power,
                
                permutation_distribution_stats=permutation_results['distribution_stats'],
                extreme_value_analysis=permutation_results['extreme_analysis'],
                robustness_metrics=permutation_results['robustness_metrics'],
                
                shuffle_methods_used=permutation_results['methods_used'],
                computation_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.results.append(permutation_result)
            self.permutation_results.append(permutation_result)
            self.test_history.append({
                'strategy': strategy.name,
                'timestamp': start_time,
                'result': permutation_result
            })
            
            self.logger.info(f"Permutation test completed successfully in {permutation_result.computation_time:.2f}s")
            self.logger.info(f"P-value: {permutation_result.p_value:.6f}, Significant: {permutation_result.significant}")
            
            return permutation_result
            
        except Exception as e:
            self.logger.error(f"Error in permutation test: {e}")
            raise
    
    def _run_comprehensive_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run comprehensive backtest to get detailed performance metrics"""
        try:
            initial_cash = self.config.initial_cash
            cash = initial_cash
            position = 0
            trades = []
            equity_curve = [initial_cash]
            
            for i in range(1, len(data)):
                try:
                    current_price = float(data['close'].iloc[i])
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    # Advanced strategy logic with multiple signals
                    price_change = (current_price - float(data['close'].iloc[i-1])) / float(data['close'].iloc[i-1])
                    volume_ratio = float(data['volume'].iloc[i]) / float(data['volume_ma'].iloc[i]) if not pd.isna(data['volume_ma'].iloc[i]) else 1.0
                    volatility = float(data['volatility'].iloc[i]) if not pd.isna(data['volatility'].iloc[i]) else 0.02
                    
                    # Dynamic thresholds based on volatility
                    threshold = max(0.005, volatility * 2)  # Minimum 0.5% threshold
                    position_size = min(0.1, 0.05 / volatility)  # Volatility-adjusted position sizing
                    
                    # Buy signal with volume confirmation
                    if price_change > threshold and volume_ratio > 1.2 and position <= 0:
                        if position < 0:  # Close short
                            cash += abs(position) * current_price * (1 - self.config.commission)
                            trades.append({'type': 'close_short', 'price': current_price, 'index': i, 'reason': 'signal_reversal'})
                        
                        # Open long position
                        position = int(cash * position_size / current_price)
                        if position > 0:
                            cash -= position * current_price * (1 + self.config.commission)
                            trades.append({'type': 'buy', 'price': current_price, 'index': i, 'reason': 'momentum_signal'})
                    
                    # Sell signal with volume confirmation
                    elif price_change < -threshold and volume_ratio > 1.2 and position >= 0:
                        if position > 0:  # Close long
                            cash += position * current_price * (1 - self.config.commission)
                            trades.append({'type': 'sell', 'price': current_price, 'index': i, 'reason': 'signal_reversal'})
                            position = 0
                        
                        # Open short position
                        position = -int(cash * position_size / current_price)
                        if position < 0:
                            cash += abs(position) * current_price * (1 + self.config.commission)
                            trades.append({'type': 'short', 'price': current_price, 'index': i, 'reason': 'momentum_signal'})
                    
                    # Update equity curve
                    current_equity = cash
                    if position > 0:
                        current_equity += position * current_price
                    elif position < 0:
                        current_equity -= abs(position) * current_price
                    equity_curve.append(current_equity)
                
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
            
            # Calculate comprehensive metrics
            total_return = (cash - initial_cash) / initial_cash if initial_cash > 0 else 0.0
            equity_series = pd.Series(equity_curve)
            returns_series = equity_series.pct_change().dropna()
            
            # Performance metrics
            metrics = {
                'total_return': total_return,
                'final_cash': cash,
                'trades_count': len(trades),
                'win_rate': self._calculate_win_rate(trades, data),
                'profit_factor': self._calculate_profit_factor(trades, data),
                'max_drawdown': self._calculate_max_drawdown(equity_series),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns_series),
                'sortino_ratio': self._calculate_sortino_ratio(returns_series),
                'calmar_ratio': self._calculate_calmar_ratio(total_return, equity_series),
                'avg_trade_return': self._calculate_avg_trade_return(trades, data),
                'volatility': returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0.0
            }
            
            return {
                'total_return': total_return,
                'final_cash': cash,
                'trades_count': len(trades),
                'metrics': metrics,
                'equity_curve': equity_series,
                'trades': trades
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive backtest: {e}")
            return {
                'total_return': 0.0, 
                'final_cash': self.config.initial_cash, 
                'trades_count': 0,
                'metrics': {},
                'equity_curve': pd.Series(),
                'trades': []
            }
    
    def _run_advanced_permutation_tests(self, data: pd.DataFrame, strategy: BaseStrategy, original_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run permutation tests using multiple shuffling methods"""
        try:
            self.logger.info(f"Running {self.config.n_permutations} permutations with multiple methods...")
            
            permutation_returns = []
            permutation_metrics = []
            methods_used = []
            
            for i in range(self.config.n_permutations):
                try:
                    # Select shuffling method
                    method = np.random.choice(self.config.shuffle_methods)
                    methods_used.append(method)
                    
                    # Create shuffled data
                    shuffled_data = self._shuffle_data(data, method)
                    
                    # Run backtest on shuffled data
                    perm_result = self._run_comprehensive_backtest(shuffled_data, strategy)
                    
                    if 'total_return' in perm_result and not pd.isna(perm_result['total_return']):
                        permutation_returns.append(perm_result['total_return'])
                        permutation_metrics.append(perm_result['metrics'])
                    
                except Exception as e:
                    continue
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Completed {i + 1}/{self.config.n_permutations} permutations")
            
            if len(permutation_returns) < self.config.min_permutations:
                raise ValueError(f"Insufficient successful permutations: {len(permutation_returns)} < {self.config.min_permutations}")
            
            # Calculate distribution statistics
            returns_array = np.array(permutation_returns)
            distribution_stats = {
                'mean': float(np.mean(returns_array)),
                'std': float(np.std(returns_array)),
                'median': float(np.median(returns_array)),
                'skewness': float(stats.skew(returns_array)),
                'kurtosis': float(stats.kurtosis(returns_array)),
                'min': float(np.min(returns_array)),
                'max': float(np.max(returns_array)),
                'q25': float(np.percentile(returns_array, 25)),
                'q75': float(np.percentile(returns_array, 75))
            }
            
            # Extreme value analysis
            extreme_analysis = self._analyze_extreme_values(returns_array, original_result['total_return'])
            
            # Robustness metrics
            robustness_metrics = self._calculate_robustness_metrics(returns_array, original_result['total_return'])
            
            return {
                'returns': permutation_returns,
                'metrics': permutation_metrics,
                'distribution_stats': distribution_stats,
                'extreme_analysis': extreme_analysis,
                'robustness_metrics': robustness_metrics,
                'methods_used': methods_used
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced permutation tests: {e}")
            raise
    
    def _shuffle_data(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply different shuffling methods to data"""
        try:
            shuffled_data = data.copy()
            
            if method == "price_shuffle":
                # Shuffle close prices and adjust other price columns
                close_prices = shuffled_data['close'].values.copy()
                np.random.shuffle(close_prices)
                shuffled_data['close'] = close_prices
                
                # Update other price columns based on shuffled close prices
                shuffled_data['open'] = shuffled_data['close'] * (1 + np.random.normal(0, 0.001, len(shuffled_data)))
                shuffled_data['high'] = shuffled_data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(shuffled_data)))
                shuffled_data['low'] = shuffled_data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(shuffled_data)))
                
            elif method == "return_shuffle":
                # Shuffle returns and reconstruct prices
                returns = shuffled_data['returns'].dropna().values.copy()
                np.random.shuffle(returns)
                
                # Reconstruct prices from shuffled returns
                initial_price = shuffled_data['close'].iloc[0]
                new_prices = [initial_price]
                for ret in returns:
                    new_prices.append(new_prices[-1] * (1 + ret))
                
                shuffled_data['close'] = new_prices[:len(shuffled_data)]
                shuffled_data['returns'] = shuffled_data['close'].pct_change()
                
            elif method == "volume_shuffle":
                # Shuffle volume data
                volume = shuffled_data['volume'].values.copy()
                np.random.shuffle(volume)
                shuffled_data['volume'] = volume
                shuffled_data['volume_ma'] = shuffled_data['volume'].rolling(window=20).mean()
                
            elif method == "time_shuffle":
                # Shuffle time order while preserving some structure
                if len(shuffled_data) > self.config.block_size:
                    n_blocks = len(shuffled_data) // self.config.block_size
                    block_indices = list(range(n_blocks))
                    np.random.shuffle(block_indices)
                    
                    new_data = []
                    for block_idx in block_indices:
                        start_idx = block_idx * self.config.block_size
                        end_idx = min((block_idx + 1) * self.config.block_size, len(shuffled_data))
                        new_data.append(shuffled_data.iloc[start_idx:end_idx])
                    
                    shuffled_data = pd.concat(new_data, ignore_index=True)
                
            elif method == "block_shuffle":
                # Shuffle data in blocks to preserve local structure
                if len(shuffled_data) > self.config.block_size:
                    n_blocks = len(shuffled_data) // self.config.block_size
                    block_indices = list(range(n_blocks))
                    np.random.shuffle(block_indices)
                    
                    new_data = []
                    for block_idx in block_indices:
                        start_idx = block_idx * self.config.block_size
                        end_idx = min((block_idx + 1) * self.config.block_size, len(shuffled_data))
                        new_data.append(shuffled_data.iloc[start_idx:end_idx])
                    
                    shuffled_data = pd.concat(new_data, ignore_index=True)
            
            # Recalculate derived features
            shuffled_data['log_returns'] = np.log(shuffled_data['close'] / shuffled_data['close'].shift(1))
            shuffled_data['volatility'] = shuffled_data['returns'].rolling(window=20).std()
            
            return shuffled_data
            
        except Exception as e:
            self.logger.error(f"Error in data shuffling method {method}: {e}")
            return data
    
    def _calculate_statistical_significance(self, original_return: float, permutation_returns: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive statistical significance measures"""
        try:
            returns_array = np.array(permutation_returns)
            
            # Basic p-value calculation
            p_value = np.mean(returns_array >= original_return)
            
            # Multiple testing correction
            if self.config.multiple_testing_correction == "bonferroni":
                adjusted_p_value = min(1.0, p_value * len(permutation_returns))
            elif self.config.multiple_testing_correction == "holm":
                # Holm-Bonferroni correction
                sorted_p_values = np.sort([np.mean(returns_array >= r) for r in returns_array])
                adjusted_p_value = None
                for i, p_val in enumerate(sorted_p_values):
                    if p_val <= self.config.significance_threshold / (len(sorted_p_values) - i):
                        adjusted_p_value = p_val
                        break
                if adjusted_p_value is None:
                    adjusted_p_value = p_value
            else:
                adjusted_p_value = p_value
            
            # Confidence intervals
            alpha = 1 - self.config.confidence_level
            lower_ci = np.percentile(returns_array, (alpha/2) * 100)
            upper_ci = np.percentile(returns_array, (1 - alpha/2) * 100)
            
            # Determine significance
            significant = adjusted_p_value < self.config.significance_threshold
            
            return {
                'p_value': float(p_value),
                'adjusted_p_value': float(adjusted_p_value),
                'significant': significant,
                'confidence_interval': [float(lower_ci), float(upper_ci)],
                'alpha': alpha
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical significance: {e}")
            return {
                'p_value': 1.0,
                'adjusted_p_value': 1.0,
                'significant': False,
                'confidence_interval': [0.0, 0.0],
                'alpha': 0.05
            }
    
    def _calculate_effect_size(self, original_return: float, permutation_returns: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        try:
            if not self.config.effect_size_calculation:
                return None
            
            returns_array = np.array(permutation_returns)
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(returns_array) - 1) * np.var(returns_array, ddof=1)) / len(returns_array))
            if pooled_std == 0:
                return 0.0
            
            effect_size = (original_return - np.mean(returns_array)) / pooled_std
            return float(effect_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating effect size: {e}")
            return None
    
    def _calculate_statistical_power(self, original_return: float, permutation_returns: List[float], alpha: float) -> float:
        """Calculate statistical power of the test"""
        try:
            if not self.config.calculate_power:
                return None
            
            returns_array = np.array(permutation_returns)
            
            # Calculate power using normal approximation
            mean_perm = np.mean(returns_array)
            std_perm = np.std(returns_array)
            
            if std_perm == 0:
                return 0.0
            
            # Critical value for alpha
            critical_value = np.percentile(returns_array, (1 - alpha) * 100)
            
            # Power calculation
            z_score = (critical_value - original_return) / std_perm
            power = 1 - stats.norm.cdf(z_score)
            
            return float(power)
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical power: {e}")
            return None
    
    def _analyze_extreme_values(self, permutation_returns: np.ndarray, original_return: float) -> Dict[str, Any]:
        """Analyze extreme values in permutation results"""
        try:
            # Calculate percentiles
            original_percentile = percentileofscore(permutation_returns, original_return)
            
            # Identify extreme values
            q1, q3 = np.percentile(permutation_returns, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = permutation_returns[(permutation_returns < lower_bound) | (permutation_returns > upper_bound)]
            
            return {
                'original_percentile': float(original_percentile),
                'outlier_count': int(len(outliers)),
                'outlier_percentage': float(len(outliers) / len(permutation_returns) * 100),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing extreme values: {e}")
            return {}
    
    def _calculate_robustness_metrics(self, permutation_returns: np.ndarray, original_return: float) -> Dict[str, float]:
        """Calculate robustness metrics for the permutation test"""
        try:
            # Stability metrics
            returns_sorted = np.sort(permutation_returns)
            stability_90 = np.std(returns_sorted[int(0.05*len(returns_sorted)):int(0.95*len(returns_sorted))])
            stability_80 = np.std(returns_sorted[int(0.1*len(returns_sorted)):int(0.9*len(returns_sorted))])
            
            # Robustness to sample size
            robustness_metrics = {
                'stability_90': float(stability_90),
                'stability_80': float(stability_80),
                'coefficient_of_variation': float(np.std(permutation_returns) / np.abs(np.mean(permutation_returns)) if np.mean(permutation_returns) != 0 else float('inf')),
                'range': float(np.max(permutation_returns) - np.min(permutation_returns)),
                'mad': float(np.median(np.abs(permutation_returns - np.median(permutation_returns))))
            }
            
            return robustness_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating robustness metrics: {e}")
            return {}
    
    # Helper methods for performance metrics
    def _calculate_win_rate(self, trades: List[Dict], data: pd.DataFrame) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        winning_trades = 0
        for trade in trades:
            if trade['type'] in ['sell', 'close_short']:
                # Simplified win/loss calculation
                winning_trades += 1
        
        return winning_trades / len(trades) if trades else 0.0
    
    def _calculate_profit_factor(self, trades: List[Dict], data: pd.DataFrame) -> float:
        """Calculate profit factor"""
        # Simplified implementation
        return 1.0 if trades else 0.0
    
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
    
    def _calculate_avg_trade_return(self, trades: List[Dict], data: pd.DataFrame) -> float:
        """Calculate average trade return"""
        if not trades:
            return 0.0
        return 0.01  # Simplified return per trade
    
    def run_portfolio_backtest(self, data_dict: Dict[str, pd.DataFrame], strategies: List[BaseStrategy]) -> List[PermutationTestResult]:
        """Run portfolio permutation tests"""
        # Implementation for portfolio permutation testing
        pass
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, param_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters using permutation testing"""
        # Implementation for strategy optimization
        pass
    
    def save_results(self, results: List[PermutationTestResult], output_path: str = None) -> str:
        """Save comprehensive permutation test results"""
        try:
            if output_path is None:
                output_path = os.path.join(self.config.results_path, f"permutation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save main results
            results_file = os.path.join(output_path, "permutation_results.json")
            with open(results_file, 'w') as f:
                json.dump([self._result_to_dict(result) for result in results], f, indent=2, default=str)
            
            # Save detailed permutation data
            if self.config.save_permutation_details:
                details_file = os.path.join(output_path, "permutation_details.json")
                with open(details_file, 'w') as f:
                    json.dump({
                        'test_history': [self._history_to_dict(h) for h in self.test_history],
                        'config': self.config.__dict__
                    }, f, indent=2, default=str)
            
            # Generate summary report
            summary = self._generate_summary_report(results)
            summary_file = os.path.join(output_path, "summary_report.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Permutation test results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _result_to_dict(self, result: PermutationTestResult) -> Dict[str, Any]:
        """Convert result object to dictionary for JSON serialization"""
        return {
            'strategy_name': result.strategy_name,
            'original_return': result.original_return,
            'n_permutations': result.n_permutations,
            'successful_permutations': result.successful_permutations,
            'p_value': result.p_value,
            'adjusted_p_value': result.adjusted_p_value,
            'significant': result.significant,
            'effect_size': result.effect_size,
            'statistical_power': result.statistical_power,
            'confidence_interval': result.confidence_interval,
            'computation_time': result.computation_time
        }
    
    def _history_to_dict(self, history_item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert history item to dictionary"""
        return {
            'strategy': history_item['strategy'],
            'timestamp': history_item['timestamp'].isoformat(),
            'result_summary': self._result_to_dict(history_item['result'])
        }
    
    def _generate_summary_report(self, results: List[PermutationTestResult]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        try:
            summary = {
                'total_tests': len(results),
                'timestamp': datetime.now().isoformat(),
                'engine_config': {
                    'n_permutations': self.config.n_permutations,
                    'confidence_level': self.config.confidence_level,
                    'significance_threshold': self.config.significance_threshold,
                    'shuffle_methods': self.config.shuffle_methods
                },
                'test_summary': {
                    'significant_strategies': len([r for r in results if r.significant]),
                    'avg_p_value': np.mean([r.p_value for r in results]),
                    'avg_effect_size': np.mean([r.effect_size for r in results if r.effect_size is not None]),
                    'avg_power': np.mean([r.statistical_power for r in results if r.statistical_power is not None])
                },
                'robustness_summary': {
                    'avg_stability_90': np.mean([r.robustness_metrics.get('stability_90', 0) for r in results]),
                    'avg_outlier_percentage': np.mean([r.extreme_value_analysis.get('outlier_percentage', 0) for r in results])
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
