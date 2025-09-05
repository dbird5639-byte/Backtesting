"""
Walk-Forward Analysis Engine

This engine provides comprehensive walk-forward analysis for both strategy and risk parameters
with in-sample/out-of-sample testing, market regime detection, and advanced visualization.
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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.base.base_engine import BaseEngine, EngineConfig, BacktestResult
from core.base.base_strategy import BaseStrategy, StrategyConfig, Signal, Trade
from core.base.base_data_handler import BaseDataHandler, DataConfig
from core.base.base_risk_manager import BaseRiskManager, RiskConfig

warnings.filterwarnings('ignore')

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    # Time windows
    train_period: int = 252  # Training period in days
    test_period: int = 63    # Test period in days
    step_size: int = 21      # Step size between windows in days
    overlap: float = 0.5     # Overlap between windows (0-1)
    
    # Parameter optimization
    strategy_param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    risk_param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Risk parameter sets
    risk_sensitive_values: Dict[str, List[float]] = field(default_factory=lambda: {
        'max_drawdown': [0.05, 0.08, 0.10],
        'var_confidence': [0.99, 0.98, 0.95],
        'kelly_fraction': [0.1, 0.15, 0.2],
        'volatility_target': [0.08, 0.10, 0.12],
        'max_position_size': [0.05, 0.08, 0.10],
        'correlation_threshold': [0.5, 0.6, 0.7],
        'rebalance_frequency': ['weekly', 'biweekly', 'monthly']
    })
    
    risk_aggressive_values: Dict[str, List[float]] = field(default_factory=lambda: {
        'max_drawdown': [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'var_confidence': [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
        'kelly_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'volatility_target': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        'max_position_size': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'correlation_threshold': [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98],
        'rebalance_frequency': ['daily', 'intraday', 'real_time', 'hourly', '4h', '6h', '12h']
    })
    
    # Market regime detection
    regime_detection_method: str = "kmeans"  # "kmeans", "markov", "changepoint"
    n_regimes: int = 4
    regime_features: List[str] = field(default_factory=lambda: [
        'volatility', 'returns', 'volume', 'correlation', 'momentum', 'mean_reversion'
    ])
    
    # Output settings
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    save_regime_mapping: bool = True

@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    # Window information
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Strategy parameters
    best_strategy_params: Dict[str, Any]
    strategy_in_sample_performance: Dict[str, float]
    strategy_out_sample_performance: Dict[str, float]
    
    # Risk parameters
    best_risk_params: Dict[str, Any]
    risk_in_sample_performance: Dict[str, float]
    risk_out_sample_performance: Dict[str, float]
    
    # Market regime
    market_regime: str
    regime_confidence: float
    
    # Performance metrics
    in_sample_return: float
    out_sample_return: float
    performance_decay: float
    consistency_score: float

@dataclass
class MarketRegimeResult:
    """Market regime detection results"""
    timestamp: datetime
    regime: str
    regime_id: int
    confidence: float
    features: Dict[str, float]
    regime_characteristics: Dict[str, Any]

class WalkForwardEngine(BaseEngine):
    """
    Comprehensive walk-forward analysis engine with parameter optimization
    and market regime detection
    """
    
    def __init__(self, config: WalkForwardConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        self.results = []
        self.regime_results = []
        self.regime_mapping = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data with regime detection features"""
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
            
            # Calculate regime detection features
            data = self._calculate_regime_features(data)
            
            self.logger.info(f"Loaded data with regime features: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for market regime detection"""
        try:
            # Basic features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Volatility features
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['volatility_60'] = data['returns'].rolling(window=60).std()
            data['volatility_252'] = data['returns'].rolling(window=252).std()
            
            # Volume features
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # Price momentum features
            data['momentum_5'] = data['close'].pct_change(5)
            data['momentum_20'] = data['close'].pct_change(20)
            data['momentum_60'] = data['close'].pct_change(60)
            
            # Mean reversion features
            data['ma_20'] = data['close'].rolling(window=20).mean()
            data['ma_60'] = data['close'].rolling(window=60).mean()
            data['price_to_ma20'] = data['close'] / data['ma_20'] - 1
            data['price_to_ma60'] = data['close'] / data['ma_60'] - 1
            
            # Correlation features (rolling correlation with market proxy)
            data['rolling_correlation'] = data['returns'].rolling(window=60).corr(
                data['returns'].rolling(window=60).mean()
            )
            
            # Volatility regime features
            data['volatility_regime'] = pd.cut(
                data['volatility_60'], 
                bins=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            # Trend strength features
            data['trend_strength'] = abs(data['momentum_60']) / data['volatility_60']
            
            # Market stress features
            data['market_stress'] = (data['volatility_60'] * data['volume_ratio']) / data['trend_strength']
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating regime features: {e}")
            return data
    
    def detect_market_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regimes throughout historical data"""
        try:
            self.logger.info("Detecting market regimes...")
            
            # Prepare features for regime detection
            feature_columns = [
                'volatility', 'returns', 'volume_ratio', 'rolling_correlation',
                'momentum_20', 'price_to_ma20', 'trend_strength', 'market_stress'
            ]
            
            # Remove rows with NaN values
            clean_data = data[feature_columns].dropna()
            
            if len(clean_data) < 100:
                raise ValueError("Insufficient data for regime detection")
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(clean_data)
            
            # Detect regimes using K-means clustering
            if self.config.regime_detection_method == "kmeans":
                kmeans = KMeans(
                    n_clusters=self.config.n_regimes, 
                    random_state=42, 
                    n_init=10
                )
                regime_labels = kmeans.fit_predict(features_scaled)
                
                # Calculate regime centers and characteristics
                regime_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
            else:
                # Alternative methods can be implemented here
                raise NotImplementedError(f"Method {self.config.regime_detection_method} not implemented")
            
            # Create regime mapping
            regime_mapping = {}
            for i, (timestamp, label) in enumerate(zip(clean_data.index, regime_labels)):
                regime_mapping[timestamp] = {
                    'regime_id': int(label),
                    'regime': f'Regime_{label}',
                    'confidence': 1.0,  # K-means gives hard assignments
                    'features': dict(zip(feature_columns, features_scaled[i])),
                    'raw_features': dict(zip(feature_columns, clean_data.iloc[i]))
                }
            
            # Analyze regime characteristics
            regime_characteristics = self._analyze_regime_characteristics(
                clean_data, regime_labels, regime_centers, feature_columns
            )
            
            # Store results
            self.regime_mapping = regime_mapping
            self.regime_characteristics = regime_characteristics
            
            self.logger.info(f"Detected {self.config.n_regimes} market regimes")
            return {
                'regime_mapping': regime_mapping,
                'regime_characteristics': regime_characteristics,
                'n_regimes': self.config.n_regimes,
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            self.logger.error(f"Error in market regime detection: {e}")
            raise
    
    def _analyze_regime_characteristics(self, data: pd.DataFrame, labels: np.ndarray, 
                                      centers: np.ndarray, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of each detected regime"""
        try:
            characteristics = {}
            
            for regime_id in range(self.config.n_regimes):
                regime_mask = labels == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Calculate regime statistics
                regime_stats = {}
                for col in feature_columns:
                    regime_stats[col] = {
                        'mean': float(regime_data[col].mean()),
                        'std': float(regime_data[col].std()),
                        'min': float(regime_data[col].min()),
                        'max': float(regime_data[col].max()),
                        'median': float(regime_data[col].median())
                    }
                
                # Regime classification based on characteristics
                regime_type = self._classify_regime(regime_stats, regime_id)
                
                characteristics[f'Regime_{regime_id}'] = {
                    'regime_id': regime_id,
                    'regime_type': regime_type,
                    'n_observations': len(regime_data),
                    'percentage_of_data': len(regime_data) / len(data) * 100,
                    'statistics': regime_stats,
                    'center': dict(zip(feature_columns, centers[regime_id]))
                }
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime characteristics: {e}")
            return {}
    
    def _classify_regime(self, regime_stats: Dict[str, Any], regime_id: int) -> str:
        """Classify regime based on its characteristics"""
        try:
            # Simple classification based on volatility and momentum
            vol_mean = regime_stats.get('volatility', {}).get('mean', 0)
            mom_mean = regime_stats.get('momentum_20', {}).get('mean', 0)
            
            if vol_mean < 0.01 and abs(mom_mean) < 0.005:
                return "Low_Volatility_Sideways"
            elif vol_mean > 0.03 and abs(mom_mean) > 0.01:
                return "High_Volatility_Trending"
            elif vol_mean < 0.015 and mom_mean > 0.005:
                return "Low_Volatility_Uptrend"
            elif vol_mean < 0.015 and mom_mean < -0.005:
                return "Low_Volatility_Downtrend"
            else:
                return f"Mixed_Regime_{regime_id}"
                
        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            return f"Unknown_Regime_{regime_id}"
    
    def run_walkforward_analysis(self, data: pd.DataFrame, strategy: BaseStrategy) -> List[WalkForwardResult]:
        """Run comprehensive walk-forward analysis"""
        try:
            self.logger.info("Starting walk-forward analysis...")
            
            # First detect market regimes
            regime_info = self.detect_market_regimes(data)
            
            # Calculate walk-forward windows
            windows = self._calculate_walkforward_windows(data)
            
            results = []
            
            for i, window in enumerate(windows):
                self.logger.info(f"Processing window {i+1}/{len(windows)}")
                
                # Get window data
                train_data = data.loc[window['train_start']:window['train_end']]
                test_data = data.loc[window['test_start']:window['test_end']]
                
                # Get market regime for this period
                regime_info_window = self._get_regime_for_period(
                    test_data.index[0], regime_info['regime_mapping']
                )
                
                # Optimize strategy parameters (in-sample)
                best_strategy_params, strategy_in_sample = self._optimize_strategy_parameters(
                    train_data, strategy
                )
                
                # Test strategy parameters (out-of-sample)
                strategy_out_sample = self._test_strategy_parameters(
                    test_data, strategy, best_strategy_params
                )
                
                # Optimize risk parameters (in-sample)
                best_risk_params, risk_in_sample = self._optimize_risk_parameters(
                    train_data, strategy, best_strategy_params
                )
                
                # Test risk parameters (out-of-sample)
                risk_out_sample = self._test_risk_parameters(
                    test_data, strategy, best_strategy_params, best_risk_params
                )
                
                # Calculate performance metrics
                in_sample_return = strategy_in_sample.get('total_return', 0)
                out_sample_return = strategy_out_sample.get('total_return', 0)
                performance_decay = in_sample_return - out_sample_return
                consistency_score = self._calculate_consistency_score(
                    strategy_in_sample, strategy_out_sample
                )
                
                # Create result
                result = WalkForwardResult(
                    window_id=i,
                    train_start=window['train_start'],
                    train_end=window['train_end'],
                    test_start=window['test_start'],
                    test_end=window['test_end'],
                    
                    best_strategy_params=best_strategy_params,
                    strategy_in_sample_performance=strategy_in_sample,
                    strategy_out_sample_performance=strategy_out_sample,
                    
                    best_risk_params=best_risk_params,
                    risk_in_sample_performance=risk_in_sample,
                    risk_out_sample_performance=risk_out_sample,
                    
                    market_regime=regime_info_window['regime'],
                    regime_confidence=regime_info_window['confidence'],
                    
                    in_sample_return=in_sample_return,
                    out_sample_return=out_sample_return,
                    performance_decay=performance_decay,
                    consistency_score=consistency_score
                )
                
                results.append(result)
            
            self.results = results
            self.logger.info(f"Walk-forward analysis completed: {len(results)} windows")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            raise
    
    def _calculate_walkforward_windows(self, data: pd.DataFrame) -> List[Dict[str, datetime]]:
        """Calculate walk-forward windows"""
        try:
            windows = []
            start_date = data.index[0]
            end_date = data.index[-1]
            
            current_start = start_date
            step_days = self.config.step_size
            
            while True:
                train_end = current_start + timedelta(days=self.config.train_period)
                test_end = train_end + timedelta(days=self.config.test_period)
                
                if test_end > end_date:
                    break
                
                windows.append({
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end
                })
                
                current_start += timedelta(days=step_days)
            
            return windows
            
        except Exception as e:
            self.logger.error(f"Error calculating walk-forward windows: {e}")
            return []
    
    def _get_regime_for_period(self, timestamp: datetime, regime_mapping: Dict) -> Dict[str, Any]:
        """Get market regime information for a specific period"""
        try:
            # Find the closest regime mapping
            if timestamp in regime_mapping:
                return regime_mapping[timestamp]
            
            # If exact match not found, find closest
            available_timestamps = list(regime_mapping.keys())
            closest_timestamp = min(available_timestamps, key=lambda x: abs((x - timestamp).total_seconds()))
            
            return regime_mapping[closest_timestamp]
            
        except Exception as e:
            self.logger.error(f"Error getting regime for period: {e}")
            return {'regime': 'Unknown', 'confidence': 0.0}
    
    def _optimize_strategy_parameters(self, data: pd.DataFrame, strategy: BaseStrategy) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize strategy parameters using in-sample data"""
        try:
            best_params = {}
            best_performance = {'total_return': -float('inf')}
            
            # Generate parameter combinations
            param_combinations = self._generate_strategy_param_combinations()
            
            for params in param_combinations:
                # Test strategy with these parameters
                performance = self._test_strategy_parameters(data, strategy, params)
                
                if performance.get('total_return', -float('inf')) > best_performance['total_return']:
                    best_performance = performance
                    best_params = params.copy()
            
            return best_params, best_performance
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {e}")
            return {}, {}
    
    def _optimize_risk_parameters(self, data: pd.DataFrame, strategy: BaseStrategy, 
                                 strategy_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize risk parameters using in-sample data"""
        try:
            best_params = {}
            best_performance = {'sharpe_ratio': -float('inf')}
            
            # Generate risk parameter combinations
            risk_combinations = self._generate_risk_param_combinations()
            
            for risk_params in risk_combinations:
                # Test risk parameters
                performance = self._test_risk_parameters(data, strategy, strategy_params, risk_params)
                
                if performance.get('sharpe_ratio', -float('inf')) > best_performance['sharpe_ratio']:
                    best_performance = performance
                    best_params = risk_params.copy()
            
            return best_params, best_performance
            
        except Exception as e:
            self.logger.error(f"Error optimizing risk parameters: {e}")
            return {}, {}
    
    def _generate_strategy_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate strategy parameter combinations for testing"""
        try:
            combinations = []
            
            # Default parameters if none specified
            if not self.config.strategy_param_ranges:
                self.config.strategy_param_ranges = {
                    'short_window': [5, 10, 15, 20],
                    'long_window': [20, 30, 40, 50],
                    'threshold': [0.01, 0.02, 0.03]
                }
            
            # Generate all combinations
            import itertools
            keys = list(self.config.strategy_param_ranges.keys())
            values = list(self.config.strategy_param_ranges.values())
            
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Error generating strategy parameter combinations: {e}")
            return [{}]
    
    def _generate_risk_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate risk parameter combinations for testing"""
        try:
            combinations = []
            
            # Combine sensitive and aggressive values
            all_risk_params = {}
            for param in self.config.risk_sensitive_values:
                all_risk_params[param] = (
                    self.config.risk_sensitive_values[param] + 
                    self.config.risk_aggressive_values[param]
                )
            
            # Generate combinations
            import itertools
            keys = list(all_risk_params.keys())
            values = list(all_risk_params.values())
            
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Error generating risk parameter combinations: {e}")
            return [{}]
    
    def _test_strategy_parameters(self, data: pd.DataFrame, strategy: BaseStrategy, 
                                params: Dict[str, Any]) -> Dict[str, float]:
        """Test strategy with specific parameters"""
        try:
            # Update strategy parameters
            strategy.parameters.update(params)
            
            # Run simple backtest
            performance = self._run_simple_backtest(data, strategy)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error testing strategy parameters: {e}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0}
    
    def _test_risk_parameters(self, data: pd.DataFrame, strategy: BaseStrategy, 
                             strategy_params: Dict[str, Any], risk_params: Dict[str, Any]) -> Dict[str, float]:
        """Test risk parameters with specific settings"""
        try:
            # Update strategy parameters
            strategy.parameters.update(strategy_params)
            
            # Run backtest with risk management
            performance = self._run_risk_managed_backtest(data, strategy, risk_params)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error testing risk parameters: {e}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0}
    
    def _run_simple_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, float]:
        """Run simple backtest for parameter testing"""
        try:
            initial_cash = 100000.0
            cash = initial_cash
            position = 0
            returns = []
            
            for i in range(1, len(data)):
                try:
                    current_price = float(data['close'].iloc[i])
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    # Simple strategy logic
                    price_change = (current_price - float(data['close'].iloc[i-1])) / float(data['close'].iloc[i-1])
                    threshold = strategy.parameters.get('threshold', 0.02)
                    
                    # Trading logic
                    if price_change > threshold and position <= 0:
                        position = int(cash * 0.1 / current_price)
                        cash -= position * current_price
                    elif price_change < -threshold and position > 0:
                        cash += position * current_price
                        position = 0
                    
                    # Calculate returns
                    current_value = cash + position * current_price
                    daily_return = (current_value - initial_cash) / initial_cash
                    returns.append(daily_return)
                    
                except Exception as e:
                    continue
            
            if not returns:
                return {'total_return': 0.0, 'sharpe_ratio': 0.0}
            
            total_return = returns[-1] if returns else 0.0
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in simple backtest: {e}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0}
    
    def _run_risk_managed_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, 
                                  risk_params: Dict[str, Any]) -> Dict[str, float]:
        """Run risk-managed backtest for risk parameter testing"""
        try:
            initial_cash = 100000.0
            cash = initial_cash
            position = 0
            returns = []
            max_dd = 0.0
            
            max_drawdown = risk_params.get('max_drawdown', 0.15)
            max_position_size = risk_params.get('max_position_size', 0.2)
            
            for i in range(1, len(data)):
                try:
                    current_price = float(data['close'].iloc[i])
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    # Check drawdown limit
                    current_value = cash + position * current_price
                    if current_value < initial_cash * (1 - max_drawdown):
                        # Close position due to drawdown limit
                        if position > 0:
                            cash += position * current_price
                            position = 0
                        continue
                    
                    # Simple strategy logic
                    price_change = (current_price - float(data['close'].iloc[i-1])) / float(data['close'].iloc[i-1])
                    threshold = strategy.parameters.get('threshold', 0.02)
                    
                    # Trading logic with position sizing
                    if price_change > threshold and position <= 0:
                        position_size = min(max_position_size, 0.1)
                        position = int(cash * position_size / current_price)
                        cash -= position * current_price
                    elif price_change < -threshold and position > 0:
                        cash += position * current_price
                        position = 0
                    
                    # Calculate returns and drawdown
                    current_value = cash + position * current_price
                    daily_return = (current_value - initial_cash) / initial_cash
                    returns.append(daily_return)
                    
                    # Update max drawdown
                    peak = max(returns) if returns else 0
                    current_dd = (peak - daily_return) / peak if peak > 0 else 0
                    max_dd = max(max_dd, current_dd)
                    
                except Exception as e:
                    continue
            
            if not returns:
                return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
            
            total_return = returns[-1] if returns else 0.0
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk-managed backtest: {e}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_consistency_score(self, in_sample: Dict[str, float], 
                                   out_sample: Dict[str, float]) -> float:
        """Calculate consistency score between in-sample and out-sample performance"""
        try:
            # Simple consistency metric based on return similarity
            in_return = in_sample.get('total_return', 0)
            out_return = out_sample.get('total_return', 0)
            
            if in_return == 0:
                return 0.0
            
            # Consistency based on how well out-of-sample performance matches in-sample
            consistency = 1 - abs(in_return - out_return) / abs(in_return)
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {e}")
            return 0.0
    
    def save_results(self, results: List[WalkForwardResult], output_path: str = None) -> str:
        """Save comprehensive walk-forward analysis results"""
        try:
            if output_path is None:
                output_path = os.path.join("./results", f"walkforward_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save main results
            results_file = os.path.join(output_path, "walkforward_results.json")
            with open(results_file, 'w') as f:
                json.dump([self._result_to_dict(result) for result in results], f, indent=2, default=str)
            
            # Save regime mapping
            if self.config.save_regime_mapping:
                regime_file = os.path.join(output_path, "regime_mapping.json")
                with open(regime_file, 'w') as f:
                    json.dump(self.regime_mapping, f, indent=2, default=str)
                
                regime_char_file = os.path.join(output_path, "regime_characteristics.json")
                with open(regime_char_file, 'w') as f:
                    json.dump(self.regime_characteristics, f, indent=2, default=str)
            
            # Generate visualizations
            if self.config.generate_visualizations:
                self._generate_walkforward_visualizations(results, output_path)
            
            self.logger.info(f"Walk-forward results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _result_to_dict(self, result: WalkForwardResult) -> Dict[str, Any]:
        """Convert result object to dictionary for JSON serialization"""
        return {
            'window_id': result.window_id,
            'train_start': result.train_start.isoformat(),
            'train_end': result.train_end.isoformat(),
            'test_start': result.test_start.isoformat(),
            'test_end': result.test_end.isoformat(),
            'best_strategy_params': result.best_strategy_params,
            'strategy_in_sample_performance': result.strategy_in_sample_performance,
            'strategy_out_sample_performance': result.strategy_out_sample_performance,
            'best_risk_params': result.best_risk_params,
            'risk_in_sample_performance': result.risk_in_sample_performance,
            'risk_out_sample_performance': result.risk_out_sample_performance,
            'market_regime': result.market_regime,
            'regime_confidence': result.regime_confidence,
            'in_sample_return': result.in_sample_return,
            'out_sample_return': result.out_sample_return,
            'performance_decay': result.performance_decay,
            'consistency_score': result.consistency_score
        }
    
    def _generate_walkforward_visualizations(self, results: List[WalkForwardResult], output_path: str):
        """Generate comprehensive visualizations for walk-forward analysis"""
        try:
            # Performance over time
            self._plot_performance_over_time(results, output_path)
            
            # Parameter evolution
            self._plot_parameter_evolution(results, output_path)
            
            # Regime analysis
            self._plot_regime_analysis(results, output_path)
            
            # Performance decay analysis
            self._plot_performance_decay(results, output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def _plot_performance_over_time(self, results: List[WalkForwardResult], output_path: str):
        """Plot performance over time"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('In-Sample vs Out-of-Sample Returns', 'Performance Metrics Over Time'),
                vertical_spacing=0.1
            )
            
            # Extract data
            dates = [r.test_start for r in results]
            in_sample_returns = [r.in_sample_return for r in results]
            out_sample_returns = [r.out_sample_return for r in results]
            consistency_scores = [r.consistency_score for r in results]
            
            # Plot returns
            fig.add_trace(
                go.Scatter(x=dates, y=in_sample_returns, name='In-Sample Returns', 
                          mode='lines+markers', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=out_sample_returns, name='Out-of-Sample Returns', 
                          mode='lines+markers', line=dict(color='red')),
                row=1, col=1
            )
            
            # Plot consistency scores
            fig.add_trace(
                go.Scatter(x=dates, y=consistency_scores, name='Consistency Score', 
                          mode='lines+markers', line=dict(color='green')),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Walk-Forward Analysis Performance Over Time',
                xaxis_title='Date',
                yaxis_title='Returns/Score',
                height=800
            )
            
            # Save plot
            plot_file = os.path.join(output_path, "performance_over_time.html")
            fig.write_html(plot_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting performance over time: {e}")
    
    def _plot_parameter_evolution(self, results: List[WalkForwardResult], output_path: str):
        """Plot parameter evolution over time"""
        try:
            # Extract strategy parameters
            dates = [r.test_start for r in results]
            
            # Get unique parameter names
            param_names = set()
            for result in results:
                param_names.update(result.best_strategy_params.keys())
            
            # Create subplots for each parameter
            n_params = len(param_names)
            if n_params == 0:
                return
            
            fig = make_subplots(
                rows=n_params, cols=1,
                subplot_titles=list(param_names),
                vertical_spacing=0.05
            )
            
            for i, param_name in enumerate(param_names):
                param_values = []
                for result in results:
                    param_values.append(result.best_strategy_params.get(param_name, 0))
                
                fig.add_trace(
                    go.Scatter(x=dates, y=param_values, name=param_name, 
                              mode='lines+markers'),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                title='Strategy Parameter Evolution Over Time',
                height=200 * n_params,
                showlegend=False
            )
            
            # Save plot
            plot_file = os.path.join(output_path, "parameter_evolution.html")
            fig.write_html(plot_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting parameter evolution: {e}")
    
    def _plot_regime_analysis(self, results: List[WalkForwardResult], output_path: str):
        """Plot regime analysis"""
        try:
            # Count regimes
            regime_counts = {}
            for result in results:
                regime = result.market_regime
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(labels=list(regime_counts.keys()), 
                                   values=list(regime_counts.values()))])
            
            fig.update_layout(title='Market Regime Distribution')
            
            # Save plot
            plot_file = os.path.join(output_path, "regime_distribution.html")
            fig.write_html(plot_file)
            
            # Create regime performance heatmap
            regime_performance = {}
            for result in results:
                regime = result.market_regime
                if regime not in regime_performance:
                    regime_performance[regime] = []
                regime_performance[regime].append(result.out_sample_return)
            
            # Calculate average performance per regime
            regime_avg_performance = {}
            for regime, returns in regime_performance.items():
                regime_avg_performance[regime] = np.mean(returns)
            
            # Create heatmap data
            regimes = list(regime_avg_performance.keys())
            performance_values = list(regime_avg_performance.values())
            
            fig = go.Figure(data=go.Heatmap(
                z=[performance_values],
                x=regimes,
                y=['Performance'],
                colorscale='RdYlGn'
            ))
            
            fig.update_layout(title='Average Performance by Market Regime')
            
            # Save plot
            plot_file = os.path.join(output_path, "regime_performance_heatmap.html")
            fig.write_html(plot_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting regime analysis: {e}")
    
    def _plot_performance_decay(self, results: List[WalkForwardResult], output_path: str):
        """Plot performance decay analysis"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Performance Decay Over Time', 'Decay vs Consistency'),
                vertical_spacing=0.1
            )
            
            # Extract data
            dates = [r.test_start for r in results]
            performance_decay = [r.performance_decay for r in results]
            consistency_scores = [r.consistency_score for r in results]
            
            # Plot performance decay
            fig.add_trace(
                go.Scatter(x=dates, y=performance_decay, name='Performance Decay', 
                          mode='lines+markers', line=dict(color='orange')),
                row=1, col=1
            )
            
            # Plot decay vs consistency
            fig.add_trace(
                go.Scatter(x=consistency_scores, y=performance_decay, name='Decay vs Consistency', 
                          mode='markers', marker=dict(color='purple')),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Performance Decay Analysis',
                height=800
            )
            
            # Save plot
            plot_file = os.path.join(output_path, "performance_decay_analysis.html")
            fig.write_html(plot_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting performance decay: {e}")


# Simple strategy class for demonstration
class SimpleMAStrategy(BaseStrategy):
    """Simple moving average strategy for testing"""
    
    def __init__(self, name: str = "SimpleMA", parameters: Dict[str, Any] = None):
        super().__init__(name)
        self.parameters = parameters or {}
        self.short_window = self.parameters.get('short_window', 10)
        self.long_window = self.parameters.get('long_window', 20)
        self.threshold = self.parameters.get('threshold', 0.02)
    
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
