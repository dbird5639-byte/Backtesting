#!/usr/bin/env python3
"""
Regime Detection Engine - Focuses on identifying market regimes and creating overlays
for existing backtest results rather than testing strategies.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import warnings
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add the current directory to Python path for imports
sys.path.append('.')

warnings.filterwarnings('ignore')

@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Regime detection settings
    detection_method: str = "kmeans"  # "kmeans", "gmm", "dbscan"
    n_regimes: int = 4
    min_regime_size: int = 50
    
    # Feature engineering
    feature_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    
    # Advanced features
    include_volume_features: bool = True
    include_momentum_features: bool = True
    include_mean_reversion_features: bool = True
    include_trend_features: bool = True
    
    # Regime classification
    regime_classification_method: str = "volatility_momentum"
    
    # Output settings
    save_regime_mapping: bool = True
    generate_regime_charts: bool = True
    save_regime_statistics: bool = True
    create_overlay_visualizations: bool = True

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
    baseline_conditions: Dict[str, Any]

@dataclass
class RegimeDetectionResult:
    """Results from regime detection"""
    data_file: str
    total_regimes: int
    regime_mapping: Dict[datetime, MarketRegime]
    regime_statistics: Dict[str, Any]
    regime_transitions: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    baseline_conditions: Dict[str, Any]

class RegimeDetectionEngine:
    """
    Regime detection engine that focuses on identifying market regimes
    and creating overlays for existing backtest results.
    """
    
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.setup_logging()
        self.regime_results = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.results_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"regime_detection_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data for regime detection"""
        try:
            data = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                self.logger.warning(f"Missing required columns in {file_path}: {[col for col in required_cols if col not in data.columns]}")
                return pd.DataFrame()
            
            # Convert to datetime index if possible
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            else:
                # Create a simple datetime index
                data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='1H')
            
            # Calculate comprehensive regime features
            data = self._calculate_regime_features(data)
            
            self.logger.info(f"Loaded data for regime detection: {len(data)} rows from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive features for regime detection"""
        try:
            # Basic price features
            data['returns'] = data['Close'].pct_change()
            data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['price_range'] = (data['High'] - data['Low']) / data['Close']
            
            # Volatility features across multiple windows
            for window in self.config.volatility_windows:
                if len(data) >= window:
                    data[f'volatility_{window}'] = data['returns'].rolling(window=window).std()
                    data[f'volatility_annualized_{window}'] = data[f'volatility_{window}'] * np.sqrt(252)
            
            # Volume features
            if self.config.include_volume_features:
                data['volume_ma'] = data['Volume'].rolling(window=20).mean()
                data['volume_ratio'] = data['Volume'] / data['volume_ma']
                data['volume_trend'] = data['Volume'].rolling(window=20).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        data[f'volume_volatility_{window}'] = data['Volume'].rolling(window=window).std()
            
            # Momentum features
            if self.config.include_momentum_features:
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        data[f'momentum_{window}'] = data['Close'].pct_change(window)
                        data[f'momentum_ma_{window}'] = data['Close'].rolling(window=window).mean()
                        data[f'momentum_std_{window}'] = data['Close'].rolling(window=window).std()
            
            # Mean reversion features
            if self.config.include_mean_reversion_features:
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        ma = data['Close'].rolling(window=window).mean()
                        data[f'price_to_ma_{window}'] = data['Close'] / ma - 1
                        data[f'ma_deviation_{window}'] = (data['Close'] - ma) / ma
            
            # Trend features
            if self.config.include_trend_features:
                for window in self.config.feature_windows:
                    if len(data) >= window:
                        # Linear trend strength
                        data[f'trend_strength_{window}'] = data['Close'].rolling(window=window).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x) if len(x) > 1 else 0
                        )
                        
                        # Trend consistency
                        data[f'trend_consistency_{window}'] = data['Close'].rolling(window=window).apply(
                            lambda x: np.sum(np.diff(x) > 0) / len(x) if len(x) > 1 else 0.5
                        )
            
            # Market stress features
            data['market_stress'] = data['volatility_20'] * data['volume_ratio'] / (abs(data['trend_strength_20']) + 1e-8)
            
            # Remove infinite and NaN values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating regime features: {e}")
            return data
    
    def detect_market_regimes(self, data: pd.DataFrame, data_file: str) -> RegimeDetectionResult:
        """Detect market regimes using the specified method"""
        try:
            self.logger.info(f"Detecting market regimes for {data_file} using {self.config.detection_method}...")
            
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
            
            # Create baseline conditions
            baseline_conditions = self._create_baseline_conditions(regime_mapping, feature_columns)
            
            result = RegimeDetectionResult(
                data_file=data_file,
                total_regimes=len(set(regime_labels)),
                regime_mapping=regime_mapping,
                regime_statistics=regime_statistics,
                regime_transitions=regime_transitions,
                feature_importance=feature_importance,
                baseline_conditions=baseline_conditions
            )
            
            self.regime_results.append(result)
            self.logger.info(f"Detected {len(set(regime_labels))} market regimes for {data_file}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market regime detection for {data_file}: {e}")
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
                
                # Create baseline conditions
                baseline_conditions = self._extract_baseline_conditions(characteristics)
                
                # Create regime object
                regime = MarketRegime(
                    regime_id=int(regime_id),
                    regime_name=f"Regime_{regime_id}",
                    regime_type=regime_type,
                    start_date=regime_timestamps[0],
                    end_date=regime_timestamps[-1],
                    duration_days=(regime_timestamps[-1] - regime_timestamps[0]).days,
                    characteristics=characteristics,
                    baseline_conditions=baseline_conditions
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
    
    def _extract_baseline_conditions(self, characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Extract baseline conditions for regime identification"""
        try:
            baseline = {
                'volatility_threshold': characteristics.get('volatility_20_mean', 0),
                'return_threshold': characteristics.get('returns_mean', 0),
                'volume_threshold': characteristics.get('volume_ratio_mean', 1),
                'trend_threshold': characteristics.get('trend_strength_20_mean', 0),
                'market_stress_threshold': characteristics.get('market_stress_mean', 0),
                'conditions': {
                    'high_volatility': characteristics.get('volatility_20_mean', 0) > 0.02,
                    'positive_momentum': characteristics.get('returns_mean', 0) > 0.001,
                    'high_volume': characteristics.get('volume_ratio_mean', 1) > 1.2,
                    'strong_trend': abs(characteristics.get('trend_strength_20_mean', 0)) > 0.001,
                    'market_stress': characteristics.get('market_stress_mean', 0) > 0.2
                }
            }
            
            return baseline
            
        except Exception as e:
            self.logger.error(f"Error extracting baseline conditions: {e}")
            return {}
    
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
                
                # Convert list of Series to DataFrame
                regime_df = pd.DataFrame(regime_list)
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
    
    def _create_baseline_conditions(self, regime_mapping: Dict[datetime, MarketRegime], 
                                  feature_columns: List[str]) -> Dict[str, Any]:
        """Create baseline conditions for regime identification"""
        try:
            baseline_conditions = {}
            
            # Group regimes by type
            regime_types = {}
            for regime in regime_mapping.values():
                if regime.regime_type not in regime_types:
                    regime_types[regime.regime_type] = []
                regime_types[regime.regime_type].append(regime)
            
            # Create baseline conditions for each regime type
            for regime_type, regimes in regime_types.items():
                if not regimes:
                    continue
                
                # Calculate average characteristics for this regime type
                avg_characteristics = {}
                for feature in feature_columns:
                    values = []
                    for regime in regimes:
                        if f'{feature}_mean' in regime.characteristics:
                            values.append(regime.characteristics[f'{feature}_mean'])
                    
                    if values:
                        avg_characteristics[feature] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                baseline_conditions[regime_type] = {
                    'avg_characteristics': avg_characteristics,
                    'n_regimes': len(regimes),
                    'avg_duration_days': np.mean([r.duration_days for r in regimes]),
                    'identification_rules': self._create_identification_rules(avg_characteristics)
                }
            
            return baseline_conditions
            
        except Exception as e:
            self.logger.error(f"Error creating baseline conditions: {e}")
            return {}
    
    def _create_identification_rules(self, characteristics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Create rules for identifying regimes based on characteristics"""
        try:
            rules = {}
            
            # Volatility rules
            if 'volatility_20' in characteristics:
                vol_mean = characteristics['volatility_20']['mean']
                vol_std = characteristics['volatility_20']['std']
                rules['volatility'] = {
                    'low': vol_mean - vol_std,
                    'high': vol_mean + vol_std,
                    'threshold': vol_mean
                }
            
            # Return rules
            if 'returns' in characteristics:
                ret_mean = characteristics['returns']['mean']
                ret_std = characteristics['returns']['std']
                rules['returns'] = {
                    'positive': ret_mean + ret_std,
                    'negative': ret_mean - ret_std,
                    'threshold': ret_mean
                }
            
            # Volume rules
            if 'volume_ratio' in characteristics:
                vol_ratio_mean = characteristics['volume_ratio']['mean']
                vol_ratio_std = characteristics['volume_ratio']['std']
                rules['volume'] = {
                    'low': vol_ratio_mean - vol_ratio_std,
                    'high': vol_ratio_mean + vol_ratio_std,
                    'threshold': vol_ratio_mean
                }
            
            # Trend rules
            if 'trend_strength_20' in characteristics:
                trend_mean = characteristics['trend_strength_20']['mean']
                trend_std = characteristics['trend_strength_20']['std']
                rules['trend'] = {
                    'uptrend': trend_mean + trend_std,
                    'downtrend': trend_mean - trend_std,
                    'threshold': trend_mean
                }
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error creating identification rules: {e}")
            return {}
    
    def create_regime_overlay(self, backtest_results: List[Dict[str, Any]], 
                            regime_result: RegimeDetectionResult) -> Dict[str, Any]:
        """Create regime overlay for existing backtest results"""
        try:
            self.logger.info("Creating regime overlay for backtest results...")
            
            regime_mapping = regime_result.regime_mapping
            
            # Analyze results by regime
            regime_performance = {}
            regime_consistency = {}
            
            for result in backtest_results:
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
                regime_type = regime.regime_type
                
                # Initialize regime performance tracking
                if regime_type not in regime_performance:
                    regime_performance[regime_type] = {
                        'returns': [],
                        'sharpe_ratios': [],
                        'max_drawdowns': [],
                        'win_rates': [],
                        'n_results': 0,
                        'regime_conditions': regime.baseline_conditions
                    }
                
                # Extract performance metrics
                if 'total_return' in result:
                    regime_performance[regime_type]['returns'].append(result['total_return'])
                
                if 'sharpe_ratio' in result:
                    regime_performance[regime_type]['sharpe_ratios'].append(result['sharpe_ratio'])
                
                if 'max_drawdown' in result:
                    regime_performance[regime_type]['max_drawdowns'].append(result['max_drawdown'])
                
                if 'win_rate' in result:
                    regime_performance[regime_type]['win_rates'].append(result['win_rate'])
                
                regime_performance[regime_type]['n_results'] += 1
            
            # Calculate regime performance statistics
            for regime_type, metrics in regime_performance.items():
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
            
            # Create comprehensive overlay analysis
            regime_overlay = {
                'regime_performance': regime_performance,
                'regime_consistency': regime_consistency,
                'regime_statistics': regime_result.regime_statistics,
                'regime_transitions': regime_result.regime_transitions,
                'feature_importance': regime_result.feature_importance,
                'baseline_conditions': regime_result.baseline_conditions,
                'overlay_summary': self._create_overlay_summary(regime_performance)
            }
            
            return regime_overlay
            
        except Exception as e:
            self.logger.error(f"Error creating regime overlay: {e}")
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
    
    def _create_overlay_summary(self, regime_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of regime overlay analysis"""
        try:
            summary = {
                'total_regimes': len(regime_performance),
                'best_performing_regime': None,
                'worst_performing_regime': None,
                'most_consistent_regime': None,
                'regime_rankings': {}
            }
            
            if not regime_performance:
                return summary
            
            # Find best and worst performing regimes
            avg_returns = {regime: metrics['avg_return'] for regime, metrics in regime_performance.items()}
            avg_sharpe = {regime: metrics['avg_sharpe'] for regime, metrics in regime_performance.items()}
            consistency = {regime: metrics['return_consistency'] for regime, metrics in regime_performance.items()}
            
            if avg_returns:
                summary['best_performing_regime'] = max(avg_returns, key=avg_returns.get)
                summary['worst_performing_regime'] = min(avg_returns, key=avg_returns.get)
            
            if consistency:
                summary['most_consistent_regime'] = max(consistency, key=consistency.get)
            
            # Create rankings
            summary['regime_rankings'] = {
                'by_return': sorted(avg_returns.items(), key=lambda x: x[1], reverse=True),
                'by_sharpe': sorted(avg_sharpe.items(), key=lambda x: x[1], reverse=True),
                'by_consistency': sorted(consistency.items(), key=lambda x: x[1], reverse=True)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating overlay summary: {e}")
            return {}
    
    def save_regime_results(self, result: RegimeDetectionResult, output_path: str = None) -> str:
        """Save regime detection results"""
        try:
            if output_path is None:
                output_path = Path(self.config.results_path) / "RegimeDetectionEngine" / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_path / "regime_detection_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'data_file': result.data_file,
                    'total_regimes': result.total_regimes,
                    'regime_statistics': result.regime_statistics,
                    'regime_transitions': result.regime_transitions,
                    'feature_importance': result.feature_importance,
                    'baseline_conditions': result.baseline_conditions
                }, f, indent=2, default=str)
            
            # Save regime mapping
            if self.config.save_regime_mapping:
                mapping_file = output_path / "regime_mapping.json"
                with open(mapping_file, 'w') as f:
                    # Convert datetime objects to strings for JSON serialization
                    mapping_data = {}
                    for timestamp, regime in result.regime_mapping.items():
                        mapping_data[timestamp.isoformat()] = {
                            'regime_id': regime.regime_id,
                            'regime_name': regime.regime_name,
                            'regime_type': regime.regime_type,
                            'start_date': regime.start_date.isoformat(),
                            'end_date': regime.end_date.isoformat(),
                            'duration_days': regime.duration_days,
                            'characteristics': regime.characteristics,
                            'baseline_conditions': regime.baseline_conditions
                        }
                    
                    json.dump(mapping_data, f, indent=2)
            
            # Save regime statistics
            if self.config.save_regime_statistics:
                stats_file = output_path / "regime_statistics.json"
                with open(stats_file, 'w') as f:
                    json.dump(result.regime_statistics, f, indent=2, default=str)
            
            # Save baseline conditions
            baseline_file = output_path / "baseline_conditions.json"
            with open(baseline_file, 'w') as f:
                json.dump(result.baseline_conditions, f, indent=2, default=str)
            
            self.logger.info(f"Regime detection results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving regime results: {e}")
            raise
    
    def save_regime_overlay(self, overlay_result: Dict[str, Any], output_path: str = None) -> str:
        """Save regime overlay results"""
        try:
            if output_path is None:
                output_path = Path(self.config.results_path) / "RegimeOverlayEngine" / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save overlay results
            overlay_file = output_path / "regime_overlay_results.json"
            with open(overlay_file, 'w') as f:
                json.dump(overlay_result, f, indent=2, default=str)
            
            # Save summary
            summary_file = output_path / "overlay_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(overlay_result.get('overlay_summary', {}), f, indent=2, default=str)
            
            self.logger.info(f"Regime overlay results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving regime overlay: {e}")
            raise
