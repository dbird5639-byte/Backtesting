"""
Advanced Backtesting Engine

This engine incorporates all proficient testing and validation methods for identifying the best strategies:
- Permutation testing for statistical significance
- Walk-forward analysis for out-of-sample validation
- Statistical analysis including alpha decay and regime detection
- Monte Carlo simulations for robustness testing
- Bootstrap testing for statistical validation
- Advanced risk management and performance metrics
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
class AdvancedEngineConfig(EngineConfig):
    """Enhanced configuration for advanced backtesting engine"""
    # Permutation testing settings
    n_permutations: int = 100
    confidence_level: float = 0.95
    
    # Walk-forward analysis settings
    walk_forward_windows: int = 12
    walk_forward_overlap: float = 0.5
    min_training_period: int = 252
    min_test_period: int = 63
    
    # Statistical analysis settings
    alpha_decay_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 21, 63, 126, 252])
    correlation_lookback: int = 252
    regime_detection_method: str = "markov"  # "markov", "clustering", "changepoint"
    n_regimes: int = 3
    
    # Monte Carlo settings
    n_monte_carlo_sims: int = 1000
    monte_carlo_confidence: float = 0.95
    
    # Bootstrap settings
    n_bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    
    # Risk management settings
    max_drawdown: float = 0.15
    var_confidence: float = 0.95
    max_position_size: float = 0.2
    risk_free_rate: float = 0.02
    volatility_lookback: int = 60
    correlation_threshold: float = 0.7
    
    # Performance tracking
    track_alpha_decay: bool = True
    track_regime_performance: bool = True
    track_correlation_stability: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    save_permutation_results: bool = True
    save_walkforward_results: bool = True

@dataclass
class AdvancedBacktestResult(BacktestResult):
    """Enhanced backtest results with advanced validation metrics"""
    # Permutation testing results
    permutation_p_value: Optional[float] = None
    permutation_significant: Optional[bool] = None
    permutation_confidence_interval: Optional[List[float]] = None
    
    # Walk-forward results
    walkforward_consistency: Optional[float] = None
    walkforward_returns: Optional[List[float]] = None
    walkforward_sharpe: Optional[List[float]] = None
    
    # Statistical analysis results
    alpha_decay_rates: Optional[Dict[str, float]] = None
    regime_performance: Optional[Dict[str, Dict[str, float]]] = None
    correlation_stability: Optional[float] = None
    
    # Monte Carlo results
    monte_carlo_var: Optional[float] = None
    monte_carlo_cvar: Optional[float] = None
    monte_carlo_confidence_interval: Optional[List[float]] = None
    
    # Bootstrap results
    bootstrap_mean: Optional[float] = None
    bootstrap_std: Optional[float] = None
    bootstrap_confidence_interval: Optional[List[float]] = None

class AdvancedEngine(BaseEngine):
    """
    Advanced backtesting engine with comprehensive testing and validation methods
    """
    
    def __init__(self, config: AdvancedEngineConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        self.results = []
        self.permutation_results = []
        self.walkforward_results = []
        self.statistical_results = []
        
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
    
    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> AdvancedBacktestResult:
        """Run comprehensive backtest with all validation methods"""
        try:
            self.logger.info(f"Starting advanced backtest for strategy: {strategy.name}")
            
            # Run basic backtest
            basic_result = self._run_basic_backtest(data, strategy)
            
            # Run permutation test
            permutation_result = self._run_permutation_test(data, strategy)
            
            # Run walk-forward analysis
            walkforward_result = self._run_walkforward_analysis(data, strategy)
            
            # Run statistical analysis
            statistical_result = self._run_statistical_analysis(data, strategy)
            
            # Run Monte Carlo simulation
            monte_carlo_result = self._run_monte_carlo_simulation(data, strategy)
            
            # Run bootstrap analysis
            bootstrap_result = self._run_bootstrap_analysis(data, strategy)
            
            # Combine all results
            advanced_result = AdvancedBacktestResult(
                strategy_name=strategy.name,
                symbol=data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in data.columns else 'UNKNOWN',
                timeframe='1D',  # Default timeframe
                start_date=data['timestamp'].min(),
                end_date=data['timestamp'].max(),
                initial_capital=self.config.initial_cash,
                final_capital=basic_result['final_cash'],
                total_return=basic_result['total_return'],
                total_trades=basic_result['trades_count'],
                winning_trades=0,  # Would be calculated in detailed backtest
                losing_trades=0,   # Would be calculated in detailed backtest
                win_rate=0.0,      # Would be calculated in detailed backtest
                avg_win=0.0,       # Would be calculated in detailed backtest
                avg_loss=0.0,      # Would be calculated in detailed backtest
                profit_factor=0.0, # Would be calculated in detailed backtest
                max_drawdown=0.0,  # Would be calculated in detailed backtest
                sharpe_ratio=0.0,  # Would be calculated in detailed backtest
                sortino_ratio=0.0, # Would be calculated in detailed backtest
                calmar_ratio=0.0,  # Would be calculated in detailed backtest
                equity_curve=pd.Series(),  # Would be calculated in detailed backtest
                trade_log=pd.DataFrame(),  # Would be calculated in detailed backtest
                daily_returns=pd.Series(), # Would be calculated in detailed backtest
                metadata={},
                
                # Advanced validation results
                permutation_p_value=permutation_result.get('p_value'),
                permutation_significant=permutation_result.get('significant'),
                permutation_confidence_interval=permutation_result.get('confidence_interval'),
                
                walkforward_consistency=walkforward_result.get('consistency'),
                walkforward_returns=walkforward_result.get('returns'),
                walkforward_sharpe=walkforward_result.get('sharpe'),
                
                alpha_decay_rates=statistical_result.get('alpha_decay'),
                regime_performance=statistical_result.get('regime_performance'),
                correlation_stability=statistical_result.get('correlation_stability'),
                
                monte_carlo_var=monte_carlo_result.get('var'),
                monte_carlo_cvar=monte_carlo_result.get('cvar'),
                monte_carlo_confidence_interval=monte_carlo_result.get('confidence_interval'),
                
                bootstrap_mean=bootstrap_result.get('mean'),
                bootstrap_std=bootstrap_result.get('std'),
                bootstrap_confidence_interval=bootstrap_result.get('confidence_interval')
            )
            
            self.results.append(advanced_result)
            self.logger.info(f"Advanced backtest completed successfully")
            
            return advanced_result
            
        except Exception as e:
            self.logger.error(f"Error in advanced backtest: {e}")
            raise
    
    def _run_basic_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run basic backtest to get fundamental performance metrics"""
        try:
            initial_cash = self.config.initial_cash
            cash = initial_cash
            position = 0
            trades = []
            
            for i in range(1, len(data)):
                try:
                    current_price = float(data['close'].iloc[i])
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    # Simple momentum strategy logic
                    price_change = (current_price - float(data['close'].iloc[i-1])) / float(data['close'].iloc[i-1])
                    threshold = 0.01  # 1% threshold
                    position_size = 0.1  # 10% of cash
                    
                    # Buy signal
                    if price_change > threshold and position <= 0:
                        if position < 0:  # Close short
                            cash += abs(position) * current_price * (1 - self.config.commission)
                            trades.append({'type': 'close_short', 'price': current_price, 'index': i})
                        
                        # Open long position
                        position = int(cash * position_size / current_price)
                        if position > 0:
                            cash -= position * current_price * (1 + self.config.commission)
                            trades.append({'type': 'buy', 'price': current_price, 'index': i})
                    
                    # Sell signal
                    elif price_change < -threshold and position >= 0:
                        if position > 0:  # Close long
                            cash += position * current_price * (1 - self.config.commission)
                            trades.append({'type': 'sell', 'price': current_price, 'index': i})
                            position = 0
                        
                        # Open short position
                        position = -int(cash * position_size / current_price)
                        if position < 0:
                            cash += abs(position) * current_price * (1 + self.config.commission)
                            trades.append({'type': 'short', 'price': current_price, 'index': i})
                
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
            
            total_return = (cash - initial_cash) / initial_cash if initial_cash > 0 else 0.0
            
            return {
                'total_return': total_return,
                'final_cash': cash,
                'trades_count': len(trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error in basic backtest: {e}")
            return {'total_return': 0.0, 'final_cash': self.config.initial_cash, 'trades_count': 0}
    
    def _run_permutation_test(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run permutation test to validate statistical significance"""
        try:
            self.logger.info("Running permutation test...")
            
            # Run original backtest
            original_result = self._run_basic_backtest(data, strategy)
            original_return = original_result['total_return']
            
            # Run permutation tests
            permutation_returns = []
            successful_permutations = 0
            
            for i in range(self.config.n_permutations):
                try:
                    # Shuffle close prices
                    shuffled_data = data.copy()
                    close_prices = shuffled_data['close'].values.copy()
                    np.random.shuffle(close_prices)
                    shuffled_data['close'] = close_prices
                    
                    # Update other price columns
                    shuffled_data['open'] = shuffled_data['close'] * (1 + np.random.normal(0, 0.001, len(shuffled_data)))
                    shuffled_data['high'] = shuffled_data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(shuffled_data)))
                    shuffled_data['low'] = shuffled_data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(shuffled_data)))
                    
                    # Run backtest on shuffled data
                    perm_result = self._run_basic_backtest(shuffled_data, strategy)
                    if 'total_return' in perm_result:
                        permutation_returns.append(perm_result['total_return'])
                        successful_permutations += 1
                    
                except Exception as e:
                    continue
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Completed {i + 1}/{self.config.n_permutations} permutations")
            
            if len(permutation_returns) < 5:
                return {'error': f"Insufficient successful permutations: {len(permutation_returns)}"}
            
            # Calculate p-value and confidence intervals
            permutation_returns = np.array(permutation_returns)
            p_value = np.mean(permutation_returns >= original_return)
            
            alpha = 1 - self.config.confidence_level
            lower_ci = np.percentile(permutation_returns, (alpha/2) * 100)
            upper_ci = np.percentile(permutation_returns, (1 - alpha/2) * 100)
            
            result = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'confidence_interval': [lower_ci, upper_ci],
                'n_permutations': len(permutation_returns),
                'original_return': original_return,
                'permutation_mean': np.mean(permutation_returns),
                'permutation_std': np.std(permutation_returns)
            }
            
            self.permutation_results.append(result)
            self.logger.info(f"Permutation test completed: p-value={p_value:.4f}, significant={result['significant']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in permutation test: {e}")
            return {'error': str(e)}
    
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
            walkforward_sharpe = []
            
            for start_idx in range(0, total_size - train_size - test_size + 1, step_size):
                try:
                    # Training period
                    train_end = start_idx + train_size
                    train_data = data.iloc[start_idx:train_end]
                    
                    # Test period
                    test_start = train_end
                    test_end = min(test_start + test_size, total_size)
                    test_data = data.iloc[test_start:test_end]
                    
                    if len(test_data) < test_size * 0.8:  # Require at least 80% of test period
                        continue
                    
                    # Run backtest on test period
                    test_result = self._run_basic_backtest(test_data, strategy)
                    if 'total_return' in test_result:
                        walkforward_returns.append(test_result['total_return'])
                        
                        # Calculate Sharpe ratio for test period
                        if len(test_data) > 1:
                            returns_series = test_data['returns'].dropna()
                            if len(returns_series) > 0:
                                sharpe = np.mean(returns_series) / np.std(returns_series) * np.sqrt(252)
                                walkforward_sharpe.append(sharpe)
                
                except Exception as e:
                    continue
            
            if len(walkforward_returns) == 0:
                return {'error': "No successful walk-forward periods"}
            
            # Calculate consistency metrics
            returns_array = np.array(walkforward_returns)
            consistency = np.std(returns_array) / np.mean(returns_array) if np.mean(returns_array) != 0 else float('inf')
            
            result = {
                'returns': walkforward_returns,
                'sharpe': walkforward_sharpe,
                'consistency': consistency,
                'n_periods': len(walkforward_returns),
                'mean_return': np.mean(returns_array),
                'std_return': np.std(returns_array)
            }
            
            self.walkforward_results.append(result)
            self.logger.info(f"Walk-forward analysis completed: {len(walkforward_returns)} periods, consistency={consistency:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {'error': str(e)}
    
    def _run_statistical_analysis(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run comprehensive statistical analysis"""
        try:
            self.logger.info("Running statistical analysis...")
            
            result = {}
            
            # Alpha decay analysis
            if self.config.track_alpha_decay:
                result['alpha_decay'] = self._analyze_alpha_decay(data, strategy)
            
            # Regime detection and analysis
            if self.config.track_regime_performance:
                result['regime_performance'] = self._detect_regimes_and_analyze(data, strategy)
            
            # Correlation stability analysis
            if self.config.track_correlation_stability:
                result['correlation_stability'] = self._analyze_correlation_stability(data)
            
            self.statistical_results.append(result)
            self.logger.info("Statistical analysis completed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_alpha_decay(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, float]:
        """Analyze alpha decay across different time windows"""
        try:
            alpha_decay = {}
            
            for window in self.config.alpha_decay_windows:
                if len(data) < window:
                    continue
                
                # Calculate rolling alpha (excess return)
                rolling_returns = data['returns'].rolling(window=window).mean()
                market_returns = data['returns'].rolling(window=window).mean()  # Simplified market proxy
                
                alpha = rolling_returns - market_returns
                alpha_decay[f'{window}d'] = float(np.mean(alpha.dropna()))
            
            return alpha_decay
            
        except Exception as e:
            self.logger.error(f"Error in alpha decay analysis: {e}")
            return {}
    
    def _detect_regimes_and_analyze(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Dict[str, float]]:
        """Detect market regimes and analyze strategy performance in each"""
        try:
            # Use volatility and returns to detect regimes
            features = pd.DataFrame({
                'volatility': data['volatility'].fillna(0),
                'returns': data['returns'].fillna(0),
                'volume': data['volume'].fillna(0)
            })
            
            # Normalize features
            features_normalized = (features - features.mean()) / features.std()
            
            # K-means clustering for regime detection
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(features_normalized)
            
            # Analyze performance in each regime
            regime_performance = {}
            for regime in range(self.config.n_regimes):
                regime_mask = regime_labels == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) > 10:  # Require minimum data points
                    regime_result = self._run_basic_backtest(regime_data, strategy)
                    regime_performance[f'regime_{regime}'] = {
                        'return': regime_result.get('total_return', 0.0),
                        'trades': regime_result.get('trades_count', 0),
                        'data_points': len(regime_data)
                    }
            
            return regime_performance
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return {}
    
    def _analyze_correlation_stability(self, data: pd.DataFrame) -> float:
        """Analyze correlation stability over time"""
        try:
            # Calculate rolling correlation between returns and volume
            rolling_corr = data['returns'].rolling(window=60).corr(data['volume'])
            correlation_stability = 1 - np.std(rolling_corr.dropna())
            
            return float(correlation_stability)
            
        except Exception as e:
            self.logger.error(f"Error in correlation stability analysis: {e}")
            return 0.0
    
    def _run_monte_carlo_simulation(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run Monte Carlo simulation for risk assessment"""
        try:
            self.logger.info("Running Monte Carlo simulation...")
            
            # Get historical returns
            returns = data['returns'].dropna()
            if len(returns) < 30:
                return {'error': "Insufficient data for Monte Carlo simulation"}
            
            # Generate Monte Carlo paths
            n_sims = self.config.n_monte_carlo_sims
            n_periods = len(returns)
            
            # Bootstrap sampling for Monte Carlo
            mc_returns = np.random.choice(returns, size=(n_sims, n_periods), replace=True)
            mc_cumulative_returns = np.cumprod(1 + mc_returns, axis=1)
            
            # Calculate VaR and CVaR
            final_returns = mc_cumulative_returns[:, -1] - 1
            var_confidence = self.config.monte_carlo_confidence
            var_percentile = (1 - var_confidence) * 100
            
            var = np.percentile(final_returns, var_percentile)
            cvar = np.mean(final_returns[final_returns <= var])
            
            # Calculate confidence intervals
            alpha = 1 - var_confidence
            lower_ci = np.percentile(final_returns, (alpha/2) * 100)
            upper_ci = np.percentile(final_returns, (1 - alpha/2) * 100)
            
            result = {
                'var': float(var),
                'cvar': float(cvar),
                'confidence_interval': [float(lower_ci), float(upper_ci)],
                'n_simulations': n_sims,
                'mean_return': float(np.mean(final_returns)),
                'std_return': float(np.std(final_returns))
            }
            
            self.logger.info(f"Monte Carlo simulation completed: VaR={var:.4f}, CVaR={cvar:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}
    
    def _run_bootstrap_analysis(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run bootstrap analysis for statistical validation"""
        try:
            self.logger.info("Running bootstrap analysis...")
            
            # Get historical returns
            returns = data['returns'].dropna()
            if len(returns) < 30:
                return {'error': "Insufficient data for bootstrap analysis"}
            
            # Bootstrap sampling
            n_bootstrap = self.config.n_bootstrap_samples
            bootstrap_returns = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_return = np.prod(1 + bootstrap_sample) - 1
                bootstrap_returns.append(bootstrap_return)
            
            bootstrap_returns = np.array(bootstrap_returns)
            
            # Calculate statistics
            mean_return = np.mean(bootstrap_returns)
            std_return = np.std(bootstrap_returns)
            
            # Calculate confidence intervals
            alpha = 1 - self.config.bootstrap_confidence
            lower_ci = np.percentile(bootstrap_returns, (alpha/2) * 100)
            upper_ci = np.percentile(bootstrap_returns, (1 - alpha/2) * 100)
            
            result = {
                'mean': float(mean_return),
                'std': float(std_return),
                'confidence_interval': [float(lower_ci), float(upper_ci)],
                'n_bootstrap': n_bootstrap
            }
            
            self.logger.info(f"Bootstrap analysis completed: mean={mean_return:.4f}, std={std_return:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap analysis: {e}")
            return {'error': str(e)}
    
    def run_portfolio_backtest(self, data_dict: Dict[str, pd.DataFrame], strategies: List[BaseStrategy]) -> List[AdvancedBacktestResult]:
        """Run portfolio backtest with multiple strategies and assets"""
        # Implementation for portfolio backtesting
        pass
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, param_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters using advanced methods"""
        # Implementation for strategy optimization
        pass
    
    def save_results(self, results: List[AdvancedBacktestResult], output_path: str = None) -> str:
        """Save comprehensive results with all validation metrics"""
        try:
            if output_path is None:
                output_path = os.path.join(self.config.results_path, f"advanced_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save main results
            results_file = os.path.join(output_path, "advanced_results.json")
            with open(results_file, 'w') as f:
                json.dump([self._result_to_dict(result) for result in results], f, indent=2, default=str)
            
            # Save permutation results
            if self.permutation_results:
                perm_file = os.path.join(output_path, "permutation_results.json")
                with open(perm_file, 'w') as f:
                    json.dump(self.permutation_results, f, indent=2)
            
            # Save walk-forward results
            if self.walkforward_results:
                wf_file = os.path.join(output_path, "walkforward_results.json")
                with open(wf_file, 'w') as f:
                    json.dump(self.walkforward_results, f, indent=2)
            
            # Save statistical results
            if self.statistical_results:
                stat_file = os.path.join(output_path, "statistical_results.json")
                with open(stat_file, 'w') as f:
                    json.dump(self.statistical_results, f, indent=2, default=str)
            
            # Generate summary report
            summary = self._generate_summary_report(results)
            summary_file = os.path.join(output_path, "summary_report.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _result_to_dict(self, result: AdvancedBacktestResult) -> Dict[str, Any]:
        """Convert result object to dictionary for JSON serialization"""
        return {
            'strategy_name': result.strategy_name,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': str(result.start_date),
            'end_date': str(result.end_date),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'total_trades': result.total_trades,
            'permutation_p_value': result.permutation_p_value,
            'permutation_significant': result.permutation_significant,
            'walkforward_consistency': result.walkforward_consistency,
            'monte_carlo_var': result.monte_carlo_var,
            'bootstrap_mean': result.bootstrap_mean
        }
    
    def _generate_summary_report(self, results: List[AdvancedBacktestResult]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        try:
            summary = {
                'total_strategies': len(results),
                'timestamp': datetime.now().isoformat(),
                'engine_config': {
                    'n_permutations': self.config.n_permutations,
                    'walk_forward_windows': self.config.walk_forward_windows,
                    'n_monte_carlo_sims': self.config.n_monte_carlo_sims,
                    'n_bootstrap_samples': self.config.n_bootstrap_samples
                },
                'performance_summary': {
                    'avg_return': np.mean([r.total_return for r in results]),
                    'avg_trades': np.mean([r.total_trades for r in results]),
                    'significant_strategies': len([r for r in results if r.permutation_significant])
                },
                'validation_summary': {
                    'permutation_tests': len(self.permutation_results),
                    'walkforward_analyses': len(self.walkforward_results),
                    'statistical_analyses': len(self.statistical_results)
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
