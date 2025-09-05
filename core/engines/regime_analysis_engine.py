"""
Market Regime Analysis Engine

This engine identifies market regimes throughout historical data and applies
regime information to backtest results from all engines, providing comprehensive
regime-based analysis and visualization.
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
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
class RegimeAnalysisConfig:
    """Configuration for market regime analysis"""
    # Regime detection settings
    detection_method: str = "kmeans"  # "kmeans", "gmm", "dbscan", "hierarchical"
    n_regimes: int = 4
    min_regime_size: int = 50
    
    # Feature engineering
    feature_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    correlation_windows: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # Advanced features
    include_volume_features: bool = True
    include_momentum_features: bool = True
    include_mean_reversion_features: bool = True
    include_trend_features: bool = True
    include_regime_transition_features: bool = True
    
    # Regime classification
    regime_classification_method: str = "volatility_momentum"  # "volatility_momentum", "trend_strength", "market_stress"
    
    # Output settings
    save_regime_mapping: bool = True
    generate_regime_charts: bool = True
    save_regime_statistics: bool = True

@dataclass
class MarketRegime:
    """Market regime information"""
    regime_id: int
    regime_name: str
    regime_type: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    characteristics: Dict[str, float]
    performance_metrics: Dict[str, float]
    transition_probability: Dict[int, float]

@dataclass
class RegimeAnalysisResult:
    """Results from regime analysis"""
    data_file: str
    total_regimes: int
    regime_mapping: Dict[datetime, MarketRegime]
    regime_statistics: Dict[str, Any]
    regime_transitions: List[Dict[str, Any]]
    feature_importance: Dict[str, float]

class RegimeAnalysisEngine(BaseEngine):
    """
    Comprehensive market regime analysis engine
    """
    
    def __init__(self, config: RegimeAnalysisConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        self.regime_results = []
        self.global_regime_mapping = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data for regime analysis"""
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
                data = data.set_index('timestamp')
            
            # Calculate comprehensive regime features
            data = self._calculate_regime_features(data)
            
            self.logger.info(f"Loaded data for regime analysis: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive features for regime detection"""
        try:
            # Basic price features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['price_range'] = (data['high'] - data['low']) / data['close']
            
            # Volatility features across multiple windows
            for window in self.config.volatility_windows:
                if len(data) >= window:
                    data[f'volatility_{window}'] = data['returns'].rolling(window=window).std()
                    data[f'volatility_annualized_{window}'] = data[f'volatility_{window}'] * np.sqrt(252)
            
            # Volume features
            if self.config.include_volume_features:
                data['volume_ma'] = data['volume'].rolling(window=20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_ma']
                data['volume_trend'] = data['volume'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
                
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        data[f'volume_volatility_{window}'] = data['volume'].rolling(window=window).std()
            
            # Momentum features
            if self.config.include_momentum_features:
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        data[f'momentum_{window}'] = data['close'].pct_change(window)
                        data[f'momentum_ma_{window}'] = data['close'].rolling(window=window).mean()
                        data[f'momentum_std_{window}'] = data['close'].rolling(window=window).std()
            
            # Mean reversion features
            if self.config.include_mean_reversion_features:
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        ma = data['close'].rolling(window=window).mean()
                        data[f'price_to_ma_{window}'] = data['close'] / ma - 1
                        data[f'ma_deviation_{window}'] = (data['close'] - ma) / ma
            
            # Trend features
            if self.config.include_trend_features:
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        # Linear trend strength
                        data[f'trend_strength_{window}'] = data['close'].rolling(window=window).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x)
                        )
                        
                        # Trend consistency
                        data[f'trend_consistency_{window}'] = data['close'].rolling(window=window).apply(
                            lambda x: np.sum(np.diff(x) > 0) / len(x) if len(x) > 1 else 0.5
                        )
            
            # Correlation features
            for window in self.config.correlation_windows:
                if len(data) >= window:
                    # Rolling correlation with market proxy (simplified)
                    data[f'correlation_{window}'] = data['returns'].rolling(window=window).corr(
                        data['returns'].rolling(window=window).mean()
                    )
            
            # Market stress features
            data['market_stress'] = data['volatility_20'] * data['volume_ratio'] / (abs(data['trend_strength_20']) + 1e-8)
            
            # Regime transition features
            if self.config.include_regime_transition_features:
                data['volatility_change'] = data['volatility_20'].pct_change()
                data['volume_change'] = data['volume_ratio'].pct_change()
                data['momentum_change'] = data['momentum_20'].pct_change()
            
            # Remove infinite and NaN values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating regime features: {e}")
            return data
    
    def detect_market_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regimes using the specified method"""
        try:
            self.logger.info(f"Detecting market regimes using {self.config.detection_method}...")
            
            # Select feature columns for regime detection
            feature_columns = self._select_regime_features(data)
            
            if len(feature_columns) == 0:
                raise ValueError("No valid features found for regime detection")
            
            # Prepare feature matrix
            feature_data = data[feature_columns].dropna()
            
            if len(feature_data) < self.config.min_regime_size * self.config.n_regimes:
                raise ValueError(f"Insufficient data for regime detection: {len(feature_data)} < {self.config.min_regime_size * self.config.n_regimes}")
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_data)
            
            # Detect regimes
            if self.config.detection_method == "kmeans":
                regime_labels = self._detect_regimes_kmeans(features_scaled)
            elif self.config.detection_method == "gmm":
                regime_labels = self._detect_regimes_gmm(features_scaled)
            elif self.config.detection_method == "dbscan":
                regime_labels = self._detect_regimes_dbscan(features_scaled)
            else:
                raise ValueError(f"Unsupported detection method: {self.config.detection_method}")
            
            # Create regime mapping
            regime_mapping = self._create_regime_mapping(
                data, feature_data, regime_labels, scaler, feature_columns
            )
            
            # Analyze regime characteristics
            regime_statistics = self._analyze_regime_characteristics(
                data, regime_mapping, feature_columns
            )
            
            # Analyze regime transitions
            regime_transitions = self._analyze_regime_transitions(regime_mapping)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                feature_data, regime_labels, feature_columns
            )
            
            results = {
                'regime_mapping': regime_mapping,
                'regime_statistics': regime_statistics,
                'regime_transitions': regime_transitions,
                'feature_importance': feature_importance,
                'detection_method': self.config.detection_method,
                'n_regimes': len(set(regime_labels)),
                'feature_columns': feature_columns
            }
            
            self.regime_results.append(results)
            self.logger.info(f"Detected {len(set(regime_labels))} market regimes")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in market regime detection: {e}")
            raise
    
    def _select_regime_features(self, data: pd.DataFrame) -> List[str]:
        """Select appropriate features for regime detection"""
        try:
            # Base features that are always included
            base_features = ['volatility_20', 'returns', 'volume_ratio']
            
            # Additional features based on configuration
            additional_features = []
            
            if self.config.include_momentum_features:
                additional_features.extend([f'momentum_{w}' for w in self.config.feature_windows if f'momentum_{w}' in data.columns])
            
            if self.config.include_mean_reversion_features:
                additional_features.extend([f'price_to_ma_{w}' for w in self.config.feature_windows if f'price_to_ma_{w}' in data.columns])
            
            if self.config.include_trend_features:
                additional_features.extend([f'trend_strength_{w}' for w in self.config.feature_windows if f'trend_strength_{w}' in data.columns])
            
            if self.config.include_volume_features:
                additional_features.extend(['volume_trend', 'volume_volatility_20'])
            
            # Combine and filter valid features
            all_features = base_features + additional_features
            valid_features = [f for f in all_features if f in data.columns and data[f].notna().sum() > len(data) * 0.8]
            
            return valid_features
            
        except Exception as e:
            self.logger.error(f"Error selecting regime features: {e}")
            return []
    
    def _detect_regimes_kmeans(self, features_scaled: np.ndarray) -> np.ndarray:
        """Detect regimes using K-means clustering"""
        try:
            kmeans = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42,
                n_init=10,
                max_iter=1000
            )
            regime_labels = kmeans.fit_predict(features_scaled)
            return regime_labels
            
        except Exception as e:
            self.logger.error(f"Error in K-means regime detection: {e}")
            raise
    
    def _detect_regimes_gmm(self, features_scaled: np.ndarray) -> np.ndarray:
        """Detect regimes using Gaussian Mixture Models"""
        try:
            gmm = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=42,
                n_init=10,
                max_iter=1000
            )
            regime_labels = gmm.fit_predict(features_scaled)
            return regime_labels
            
        except Exception as e:
            self.logger.error(f"Error in GMM regime detection: {e}")
            raise
    
    def _detect_regimes_dbscan(self, features_scaled: np.ndarray) -> np.ndarray:
        """Detect regimes using DBSCAN clustering"""
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            regime_labels = dbscan.fit_predict(features_scaled)
            
            # DBSCAN may assign -1 to noise points, convert to positive regime IDs
            unique_labels = set(regime_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            # Reassign noise points to nearest regime
            for i, label in enumerate(regime_labels):
                if label == -1:
                    # Find nearest non-noise point
                    distances = np.linalg.norm(features_scaled[i] - features_scaled[regime_labels != -1], axis=1)
                    nearest_idx = np.argmin(distances)
                    regime_labels[i] = regime_labels[regime_labels != -1][nearest_idx]
            
            return regime_labels
            
        except Exception as e:
            self.logger.error(f"Error in DBSCAN regime detection: {e}")
            raise
    
    def _create_regime_mapping(self, data: pd.DataFrame, feature_data: pd.DataFrame, 
                              regime_labels: np.ndarray, scaler: StandardScaler, 
                              feature_columns: List[str]) -> Dict[datetime, MarketRegime]:
        """Create comprehensive regime mapping"""
        try:
            regime_mapping = {}
            
            # Group data by regime
            regime_groups = {}
            for i, label in enumerate(regime_labels):
                if label not in regime_groups:
                    regime_groups[label] = []
                regime_groups[label].append(i)
            
            # Create regime objects
            for regime_id, indices in regime_groups.items():
                if len(indices) < self.config.min_regime_size:
                    continue
                
                # Get regime data
                regime_data = feature_data.iloc[indices]
                regime_timestamps = regime_data.index
                
                # Calculate regime characteristics
                characteristics = self._calculate_regime_characteristics(regime_data, feature_columns)
                
                # Classify regime type
                regime_type = self._classify_regime_type(characteristics)
                
                # Create regime object
                regime = MarketRegime(
                    regime_id=int(regime_id),
                    regime_name=f"Regime_{regime_id}",
                    regime_type=regime_type,
                    start_date=regime_timestamps[0],
                    end_date=regime_timestamps[-1],
                    duration_days=(regime_timestamps[-1] - regime_timestamps[0]).days,
                    characteristics=characteristics,
                    performance_metrics={},
                    transition_probability={}
                )
                
                # Add to mapping
                for timestamp in regime_timestamps:
                    regime_mapping[timestamp] = regime
            
            return regime_mapping
            
        except Exception as e:
            self.logger.error(f"Error creating regime mapping: {e}")
            return {}
    
    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame, 
                                        feature_columns: List[str]) -> Dict[str, float]:
        """Calculate characteristics for a specific regime"""
        try:
            characteristics = {}
            
            for col in feature_columns:
                if col in regime_data.columns:
                    characteristics[f'{col}_mean'] = float(regime_data[col].mean())
                    characteristics[f'{col}_std'] = float(regime_data[col].std())
                    characteristics[f'{col}_median'] = float(regime_data[col].median())
                    characteristics[f'{col}_min'] = float(regime_data[col].min())
                    characteristics[f'{col}_max'] = float(regime_data[col].max())
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime characteristics: {e}")
            return {}
    
    def _classify_regime_type(self, characteristics: Dict[str, float]) -> str:
        """Classify regime type based on characteristics"""
        try:
            # Extract key metrics
            volatility = characteristics.get('volatility_20_mean', 0)
            returns = characteristics.get('returns_mean', 0)
            volume_ratio = characteristics.get('volume_ratio_mean', 1)
            trend_strength = characteristics.get('trend_strength_20_mean', 0)
            
            # Classify based on configuration
            if self.config.regime_classification_method == "volatility_momentum":
                if volatility < 0.01 and abs(returns) < 0.005:
                    return "Low_Volatility_Sideways"
                elif volatility > 0.03 and abs(returns) > 0.01:
                    return "High_Volatility_Trending"
                elif volatility < 0.015 and returns > 0.005:
                    return "Low_Volatility_Uptrend"
                elif volatility < 0.015 and returns < -0.005:
                    return "Low_Volatility_Downtrend"
                else:
                    return "Mixed_Regime"
            
            elif self.config.regime_classification_method == "trend_strength":
                if abs(trend_strength) < 0.001:
                    return "Sideways_Market"
                elif trend_strength > 0.001:
                    return "Uptrend_Market"
                else:
                    return "Downtrend_Market"
            
            elif self.config.regime_classification_method == "market_stress":
                market_stress = volatility * volume_ratio / (abs(trend_strength) + 1e-8)
                if market_stress < 0.1:
                    return "Low_Stress_Stable"
                elif market_stress < 0.3:
                    return "Medium_Stress_Volatile"
                else:
                    return "High_Stress_Crisis"
            
            else:
                return "Unknown_Regime"
                
        except Exception as e:
            self.logger.error(f"Error classifying regime type: {e}")
            return "Unknown_Regime"
    
    def _analyze_regime_characteristics(self, data: pd.DataFrame, 
                                      regime_mapping: Dict[datetime, MarketRegime],
                                      feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze overall regime characteristics"""
        try:
            # Group data by regime
            regime_data = {}
            for timestamp, regime in regime_mapping.items():
                if regime.regime_id not in regime_data:
                    regime_data[regime.regime_id] = []
                regime_data[regime.regime_id].append(data.loc[timestamp])
            
            # Calculate statistics for each regime
            regime_stats = {}
            for regime_id, regime_list in regime_data.items():
                if len(regime_list) == 0:
                    continue
                
                regime_df = pd.concat(regime_list)
                regime_stats[f'regime_{regime_id}'] = {
                    'n_observations': len(regime_df),
                    'percentage_of_data': len(regime_df) / len(data) * 100,
                    'avg_duration_days': np.mean([r.duration_days for r in regime_mapping.values() if r.regime_id == regime_id]),
                    'characteristics': {}
                }
                
                # Calculate average characteristics
                for col in feature_columns:
                    if col in regime_df.columns:
                        regime_stats[f'regime_{regime_id}']['characteristics'][col] = {
                            'mean': float(regime_df[col].mean()),
                            'std': float(regime_df[col].std())
                        }
            
            return regime_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime characteristics: {e}")
            return {}
    
    def _analyze_regime_transitions(self, regime_mapping: Dict[datetime, MarketRegime]) -> List[Dict[str, Any]]:
        """Analyze transitions between regimes"""
        try:
            transitions = []
            timestamps = sorted(regime_mapping.keys())
            
            for i in range(1, len(timestamps)):
                current_regime = regime_mapping[timestamps[i]]
                previous_regime = regime_mapping[timestamps[i-1]]
                
                if current_regime.regime_id != previous_regime.regime_id:
                    transition = {
                        'from_regime': previous_regime.regime_id,
                        'to_regime': current_regime.regime_id,
                        'transition_date': timestamps[i],
                        'from_regime_type': previous_regime.regime_type,
                        'to_regime_type': current_regime.regime_type,
                        'transition_duration': (timestamps[i] - timestamps[i-1]).days
                    }
                    transitions.append(transition)
            
            return transitions
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {e}")
            return []
    
    def _calculate_feature_importance(self, feature_data: pd.DataFrame, 
                                    regime_labels: np.ndarray, 
                                    feature_columns: List[str]) -> Dict[str, float]:
        """Calculate feature importance for regime detection"""
        try:
            feature_importance = {}
            
            for col in feature_columns:
                if col in feature_data.columns:
                    # Calculate F-statistic (variance between regimes / variance within regimes)
                    unique_regimes = set(regime_labels)
                    if len(unique_regimes) < 2:
                        feature_importance[col] = 0.0
                        continue
                    
                    # Calculate between-group variance
                    overall_mean = feature_data[col].mean()
                    between_group_var = 0
                    within_group_var = 0
                    total_n = len(feature_data)
                    
                    for regime_id in unique_regimes:
                        regime_mask = regime_labels == regime_id
                        regime_data = feature_data[col][regime_mask]
                        regime_mean = regime_data.mean()
                        regime_n = len(regime_data)
                        
                        between_group_var += regime_n * (regime_mean - overall_mean) ** 2
                        within_group_var += ((regime_data - regime_mean) ** 2).sum()
                    
                    between_group_var /= (len(unique_regimes) - 1)
                    within_group_var /= (total_n - len(unique_regimes))
                    
                    if within_group_var > 0:
                        f_stat = between_group_var / within_group_var
                        feature_importance[col] = float(f_stat)
                    else:
                        feature_importance[col] = 0.0
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def apply_regime_analysis_to_results(self, engine_results: List[Dict[str, Any]], 
                                       regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime analysis to results from other engines"""
        try:
            self.logger.info("Applying regime analysis to engine results...")
            
            regime_mapping = regime_results['regime_mapping']
            
            # Analyze results by regime
            regime_performance = {}
            regime_consistency = {}
            
            for result in engine_results:
                # Extract timestamp and performance metrics
                if 'timestamp' in result:
                    timestamp = pd.to_datetime(result['timestamp'])
                elif 'start_date' in result:
                    timestamp = pd.to_datetime(result['start_date'])
                else:
                    continue
                
                # Find corresponding regime
                regime = self._find_regime_for_timestamp(timestamp, regime_mapping)
                if regime is None:
                    continue
                
                regime_id = regime.regime_id
                
                # Initialize regime performance tracking
                if regime_id not in regime_performance:
                    regime_performance[regime_id] = {
                        'returns': [],
                        'sharpe_ratios': [],
                        'max_drawdowns': [],
                        'win_rates': [],
                        'n_results': 0
                    }
                
                # Extract performance metrics
                if 'total_return' in result:
                    regime_performance[regime_id]['returns'].append(result['total_return'])
                
                if 'sharpe_ratio' in result:
                    regime_performance[regime_id]['sharpe_ratios'].append(result['sharpe_ratio'])
                
                if 'max_drawdown' in result:
                    regime_performance[regime_id]['max_drawdowns'].append(result['max_drawdown'])
                
                if 'win_rate' in result:
                    regime_performance[regime_id]['win_rates'].append(result['win_rate'])
                
                regime_performance[regime_id]['n_results'] += 1
            
            # Calculate regime performance statistics
            for regime_id, metrics in regime_performance.items():
                if metrics['n_results'] > 0:
                    metrics['avg_return'] = np.mean(metrics['returns']) if metrics['returns'] else 0.0
                    metrics['avg_sharpe'] = np.mean(metrics['sharpe_ratios']) if metrics['sharpe_ratios'] else 0.0
                    metrics['avg_drawdown'] = np.mean(metrics['max_drawdowns']) if metrics['max_drawdowns'] else 0.0
                    metrics['avg_win_rate'] = np.mean(metrics['win_rates']) if metrics['win_rates'] else 0.0
                    
                    # Calculate consistency metrics
                    if metrics['returns']:
                        metrics['return_consistency'] = 1 - (np.std(metrics['returns']) / (abs(np.mean(metrics['returns'])) + 1e-8))
                    else:
                        metrics['return_consistency'] = 0.0
            
            # Create comprehensive analysis
            regime_analysis = {
                'regime_performance': regime_performance,
                'regime_consistency': regime_consistency,
                'regime_statistics': regime_results['regime_statistics'],
                'regime_transitions': regime_results['regime_transitions'],
                'feature_importance': regime_results['feature_importance']
            }
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error applying regime analysis to results: {e}")
            return {}
    
    def _find_regime_for_timestamp(self, timestamp: datetime, 
                                  regime_mapping: Dict[datetime, MarketRegime]) -> Optional[MarketRegime]:
        """Find the regime for a specific timestamp"""
        try:
            if timestamp in regime_mapping:
                return regime_mapping[timestamp]
            
            # If exact match not found, find closest
            available_timestamps = list(regime_mapping.keys())
            if not available_timestamps:
                return None
            
            closest_timestamp = min(available_timestamps, key=lambda x: abs((x - timestamp).total_seconds()))
            return regime_mapping[closest_timestamp]
            
        except Exception as e:
            self.logger.error(f"Error finding regime for timestamp: {e}")
            return None
    
    def generate_regime_visualizations(self, regime_results: Dict[str, Any], 
                                     regime_analysis: Dict[str, Any] = None,
                                     output_path: str = None) -> str:
        """Generate comprehensive regime visualizations"""
        try:
            if not self.config.generate_regime_charts:
                return ""
            
            if output_path is None:
                output_path = os.path.join("./results", f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Generate regime distribution chart
            self._plot_regime_distribution(regime_results, output_path)
            
            # Generate regime characteristics heatmap
            self._plot_regime_characteristics(regime_results, output_path)
            
            # Generate regime transitions chart
            self._plot_regime_transitions(regime_results, output_path)
            
            # Generate feature importance chart
            self._plot_feature_importance(regime_results, output_path)
            
            # Generate regime performance analysis if available
            if regime_analysis:
                self._plot_regime_performance(regime_analysis, output_path)
                self._plot_regime_consistency(regime_analysis, output_path)
            
            self.logger.info(f"Regime visualizations saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating regime visualizations: {e}")
            return ""
    
    def _plot_regime_distribution(self, regime_results: Dict[str, Any], output_path: str):
        """Plot regime distribution"""
        try:
            regime_stats = regime_results['regime_statistics']
            
            # Extract regime information
            regime_names = []
            regime_percentages = []
            
            for regime_key, stats in regime_stats.items():
                regime_names.append(regime_key)
                regime_percentages.append(stats['percentage_of_data'])
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(labels=regime_names, values=regime_percentages)])
            fig.update_layout(title='Market Regime Distribution')
            
            # Save chart
            chart_file = os.path.join(output_path, "regime_distribution.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting regime distribution: {e}")
    
    def _plot_regime_characteristics(self, regime_results: Dict[str, Any], output_path: str):
        """Plot regime characteristics heatmap"""
        try:
            regime_stats = regime_results['regime_statistics']
            feature_importance = regime_results['feature_importance']
            
            # Prepare data for heatmap
            feature_names = list(feature_importance.keys())
            regime_names = list(regime_stats.keys())
            
            # Create characteristic matrix
            char_matrix = []
            for regime_name in regime_names:
                regime_row = []
                for feature_name in feature_names:
                    if feature_name in regime_stats[regime_name]['characteristics']:
                        char_matrix.append([
                            regime_name,
                            feature_name,
                            regime_stats[regime_name]['characteristics'][feature_name]['mean']
                        ])
            
            if char_matrix:
                char_df = pd.DataFrame(char_matrix, columns=['Regime', 'Feature', 'Value'])
                char_pivot = char_df.pivot(index='Regime', columns='Feature', values='Value')
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=char_pivot.values,
                    x=char_pivot.columns,
                    y=char_pivot.index,
                    colorscale='RdYlBu'
                ))
                
                fig.update_layout(
                    title='Regime Characteristics Heatmap',
                    xaxis_title='Features',
                    yaxis_title='Regimes'
                )
                
                # Save chart
                chart_file = os.path.join(output_path, "regime_characteristics_heatmap.html")
                fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting regime characteristics: {e}")
    
    def _plot_regime_transitions(self, regime_results: Dict[str, Any], output_path: str):
        """Plot regime transitions"""
        try:
            transitions = regime_results['regime_transitions']
            
            if not transitions:
                return
            
            # Create transition matrix
            unique_regimes = set()
            for transition in transitions:
                unique_regimes.add(transition['from_regime'])
                unique_regimes.add(transition['to_regime'])
            
            transition_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))
            regime_id_map = {regime_id: i for i, regime_id in enumerate(sorted(unique_regimes))}
            
            for transition in transitions:
                from_idx = regime_id_map[transition['from_regime']]
                to_idx = regime_id_map[transition['to_regime']]
                transition_matrix[from_idx, to_idx] += 1
            
            # Create heatmap
            regime_labels = [f"Regime_{rid}" for rid in sorted(unique_regimes)]
            
            fig = go.Figure(data=go.Heatmap(
                z=transition_matrix,
                x=regime_labels,
                y=regime_labels,
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title='Regime Transition Matrix',
                xaxis_title='To Regime',
                yaxis_title='From Regime'
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "regime_transitions.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting regime transitions: {e}")
    
    def _plot_feature_importance(self, regime_results: Dict[str, Any], output_path: str):
        """Plot feature importance"""
        try:
            feature_importance = regime_results['feature_importance']
            
            if not feature_importance:
                return
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            # Create bar chart
            fig = go.Figure(data=go.Bar(x=feature_names, y=importance_values))
            fig.update_layout(
                title='Feature Importance for Regime Detection',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                xaxis_tickangle=-45
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "feature_importance.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
    
    def _plot_regime_performance(self, regime_analysis: Dict[str, Any], output_path: str):
        """Plot regime performance analysis"""
        try:
            regime_performance = regime_analysis['regime_performance']
            
            if not regime_performance:
                return
            
            # Create performance comparison chart
            regimes = list(regime_performance.keys())
            avg_returns = [regime_performance[r]['avg_return'] for r in regimes]
            avg_sharpe = [regime_performance[r]['avg_sharpe'] for r in regimes]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Average Returns by Regime', 'Average Sharpe Ratio by Regime'),
                vertical_spacing=0.1
            )
            
            # Returns chart
            fig.add_trace(
                go.Bar(x=[f"Regime_{r}" for r in regimes], y=avg_returns, name='Returns'),
                row=1, col=1
            )
            
            # Sharpe ratio chart
            fig.add_trace(
                go.Bar(x=[f"Regime_{r}" for r in regimes], y=avg_sharpe, name='Sharpe Ratio'),
                row=2, col=1
            )
            
            fig.update_layout(height=800, title='Regime Performance Analysis')
            
            # Save chart
            chart_file = os.path.join(output_path, "regime_performance.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting regime performance: {e}")
    
    def _plot_regime_consistency(self, regime_analysis: Dict[str, Any], output_path: str):
        """Plot regime consistency analysis"""
        try:
            regime_performance = regime_analysis['regime_performance']
            
            if not regime_performance:
                return
            
            # Create consistency chart
            regimes = list(regime_performance.keys())
            consistency_scores = [regime_performance[r]['return_consistency'] for r in regimes]
            
            fig = go.Figure(data=go.Bar(x=[f"Regime_{r}" for r in regimes], y=consistency_scores))
            fig.update_layout(
                title='Regime Performance Consistency',
                xaxis_title='Regimes',
                yaxis_title='Consistency Score'
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "regime_consistency.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting regime consistency: {e}")
    
    def save_results(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save regime analysis results"""
        try:
            if output_path is None:
                output_path = os.path.join("./results", f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save main results
            results_file = os.path.join(output_path, "regime_analysis_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save regime mapping
            if self.config.save_regime_mapping:
                mapping_file = os.path.join(output_path, "regime_mapping.json")
                with open(mapping_file, 'w') as f:
                    # Convert datetime objects to strings for JSON serialization
                    mapping_data = {}
                    for timestamp, regime in results['regime_mapping'].items():
                        mapping_data[timestamp.isoformat()] = {
                            'regime_id': regime.regime_id,
                            'regime_name': regime.regime_name,
                            'regime_type': regime.regime_type,
                            'start_date': regime.start_date.isoformat(),
                            'end_date': regime.end_date.isoformat(),
                            'duration_days': regime.duration_days,
                            'characteristics': regime.characteristics
                        }
                    
                    json.dump(mapping_data, f, indent=2)
            
            # Save regime statistics
            if self.config.save_regime_statistics:
                stats_file = os.path.join(output_path, "regime_statistics.json")
                with open(stats_file, 'w') as f:
                    json.dump(results['regime_statistics'], f, indent=2, default=str)
            
            self.logger.info(f"Regime analysis results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise


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
