#!/usr/bin/env python3
"""
Optimized Combined Regime Engine with Integrated Visualization
Combines regime detection, overlay, and visualization functionality
"""

import asyncio
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import warnings
warnings.filterwarnings('ignore')

# ML imports for regime detection
try:
    from sklearn.cluster import KMeans, GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .base_optimized_engine import BaseOptimizedEngine, BaseEngineConfig

class RegimeDetectionMethod(Enum):
    """Regime detection method enumeration"""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    HIDDEN_MARKOV = "hidden_markov"
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM_BASED = "momentum_based"
    COMBINED = "combined"

class RegimeFeature(Enum):
    """Regime feature enumeration"""
    RETURNS = "returns"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TECHNICAL_INDICATORS = "technical_indicators"
    CORRELATION = "correlation"
    MOMENTUM = "momentum"

@dataclass
class RegimeEngineConfig(BaseEngineConfig):
    """Configuration for Combined Regime Engine"""
    # Regime detection settings
    detection_method: RegimeDetectionMethod = RegimeDetectionMethod.KMEANS
    n_regimes: int = 3
    features: List[RegimeFeature] = field(default_factory=lambda: [
        RegimeFeature.RETURNS,
        RegimeFeature.VOLATILITY,
        RegimeFeature.TECHNICAL_INDICATORS
    ])
    
    # Clustering parameters
    min_regime_duration: int = 5  # minimum days in a regime
    regime_stability_threshold: float = 0.7
    
    # Feature engineering
    lookback_window: int = 20
    volatility_window: int = 20
    momentum_window: int = 10
    
    # Visualization settings
    generate_plots: bool = True
    plot_style: str = "seaborn-v0_8"
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    regime_colors: List[str] = field(default_factory=lambda: [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'
    ])

class OptimizedRegimeEngine(BaseOptimizedEngine):
    """Optimized Combined Regime Engine with integrated visualization"""
    
    def __init__(self, config: RegimeEngineConfig = None):
        super().__init__(config or RegimeEngineConfig())
        self.config: RegimeEngineConfig = self.config
        self.regime_models = {}
        self.regime_features = {}
        self.regime_history = []
        
        # Setup visualization
        if self.config.generate_plots:
            plt.style.use(self.config.plot_style)
            sns.set_palette("husl")
    
    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Returns
            if RegimeFeature.RETURNS in self.config.features:
                features['returns'] = data['Close'].pct_change()
                features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                features['abs_returns'] = np.abs(features['returns'])
            
            # Volatility
            if RegimeFeature.VOLATILITY in self.config.features:
                features['volatility'] = data['Close'].rolling(self.config.volatility_window).std()
                features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
                features['high_low_vol'] = (data['High'] - data['Low']) / data['Close']
            
            # Volume
            if RegimeFeature.VOLUME in self.config.features and 'Volume' in data.columns:
                features['volume'] = data['Volume']
                features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
                features['volume_price'] = data['Volume'] * data['Close']
            
            # Technical indicators
            if RegimeFeature.TECHNICAL_INDICATORS in self.config.features:
                features['sma_ratio'] = data['Close'] / data['Close'].rolling(20).mean()
                features['rsi'] = self._calculate_rsi(data['Close'])
                features['macd'] = self._calculate_macd(data['Close'])
                features['bb_position'] = self._calculate_bb_position(data)
            
            # Correlation
            if RegimeFeature.CORRELATION in self.config.features:
                features['price_momentum'] = data['Close'] / data['Close'].shift(5)
                features['volume_momentum'] = data['Volume'] / data['Volume'].shift(5) if 'Volume' in data.columns else 1
            
            # Momentum
            if RegimeFeature.MOMENTUM in self.config.features:
                features['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
                features['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
                features['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Error creating regime features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(0, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series(0, index=prices.index)
    
    def _calculate_bb_position(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position"""
        try:
            sma = data['Close'].rolling(period).mean()
            std = data['Close'].rolling(period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = (data['Close'] - lower_band) / (upper_band - lower_band)
            return bb_position
        except:
            return pd.Series(0.5, index=data.index)
    
    def detect_regimes(self, features: pd.DataFrame, method: RegimeDetectionMethod = None) -> Dict[str, Any]:
        """Detect market regimes"""
        method = method or self.config.detection_method
        
        try:
            if method == RegimeDetectionMethod.KMEANS:
                return self._detect_regimes_kmeans(features)
            elif method == RegimeDetectionMethod.GAUSSIAN_MIXTURE:
                return self._detect_regimes_gmm(features)
            elif method == RegimeDetectionMethod.VOLATILITY_BASED:
                return self._detect_regimes_volatility(features)
            elif method == RegimeDetectionMethod.MOMENTUM_BASED:
                return self._detect_regimes_momentum(features)
            elif method == RegimeDetectionMethod.COMBINED:
                return self._detect_regimes_combined(features)
            else:
                return self._detect_regimes_kmeans(features)
                
        except Exception as e:
            self.logger.error(f"Error detecting regimes: {e}")
            return {'success': False, 'error': str(e)}
    
    def _detect_regimes_kmeans(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using K-means clustering"""
        if not ML_AVAILABLE:
            return {'success': False, 'error': 'ML libraries not available'}
        
        try:
            # Prepare data
            X = features.fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Find optimal number of clusters
            best_n_clusters = self.config.n_regimes
            best_silhouette = -1
            
            for n_clusters in range(2, min(8, len(features) // 10)):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_clusters = n_clusters
            
            # Final clustering
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Create regime series
            regime_series = pd.Series(cluster_labels, index=features.index)
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(features, regime_series)
            
            return {
                'success': True,
                'regimes': regime_series,
                'n_regimes': best_n_clusters,
                'silhouette_score': best_silhouette,
                'regime_stats': regime_stats,
                'model': kmeans,
                'scaler': scaler,
                'method': 'kmeans'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_regimes_gmm(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using Gaussian Mixture Model"""
        if not ML_AVAILABLE:
            return {'success': False, 'error': 'ML libraries not available'}
        
        try:
            # Prepare data
            X = features.fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=self.config.n_regimes, random_state=42)
            gmm.fit(X_scaled)
            
            # Predict regimes
            cluster_labels = gmm.predict(X_scaled)
            regime_series = pd.Series(cluster_labels, index=features.index)
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(features, regime_series)
            
            return {
                'success': True,
                'regimes': regime_series,
                'n_regimes': self.config.n_regimes,
                'aic': gmm.aic(X_scaled),
                'bic': gmm.bic(X_scaled),
                'regime_stats': regime_stats,
                'model': gmm,
                'scaler': scaler,
                'method': 'gaussian_mixture'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_regimes_volatility(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes based on volatility"""
        try:
            if 'volatility' not in features.columns:
                return {'success': False, 'error': 'Volatility feature not available'}
            
            volatility = features['volatility'].fillna(method='ffill')
            
            # Define volatility thresholds
            vol_25 = volatility.quantile(0.25)
            vol_75 = volatility.quantile(0.75)
            
            # Create regimes
            regimes = pd.Series(1, index=features.index)  # Normal regime
            regimes[volatility < vol_25] = 0  # Low volatility
            regimes[volatility > vol_75] = 2  # High volatility
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(features, regimes)
            
            return {
                'success': True,
                'regimes': regimes,
                'n_regimes': 3,
                'volatility_thresholds': {'low': vol_25, 'high': vol_75},
                'regime_stats': regime_stats,
                'method': 'volatility_based'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_regimes_momentum(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes based on momentum"""
        try:
            if 'momentum_20' not in features.columns:
                return {'success': False, 'error': 'Momentum feature not available'}
            
            momentum = features['momentum_20'].fillna(0)
            
            # Define momentum thresholds
            mom_25 = momentum.quantile(0.25)
            mom_75 = momentum.quantile(0.75)
            
            # Create regimes
            regimes = pd.Series(1, index=features.index)  # Normal regime
            regimes[momentum < mom_25] = 0  # Bearish
            regimes[momentum > mom_75] = 2  # Bullish
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(features, regimes)
            
            return {
                'success': True,
                'regimes': regimes,
                'n_regimes': 3,
                'momentum_thresholds': {'bearish': mom_25, 'bullish': mom_75},
                'regime_stats': regime_stats,
                'method': 'momentum_based'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_regimes_combined(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using combined approach"""
        try:
            # Get individual regime detections
            kmeans_result = self._detect_regimes_kmeans(features)
            volatility_result = self._detect_regimes_volatility(features)
            momentum_result = self._detect_regimes_momentum(features)
            
            if not all(r.get('success', False) for r in [kmeans_result, volatility_result, momentum_result]):
                return {'success': False, 'error': 'Failed to detect regimes with combined method'}
            
            # Combine regimes (simple majority voting)
            combined_regimes = pd.DataFrame({
                'kmeans': kmeans_result['regimes'],
                'volatility': volatility_result['regimes'],
                'momentum': momentum_result['regimes']
            })
            
            # Majority vote
            final_regimes = combined_regimes.mode(axis=1)[0]
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(features, final_regimes)
            
            return {
                'success': True,
                'regimes': final_regimes,
                'n_regimes': final_regimes.nunique(),
                'individual_results': {
                    'kmeans': kmeans_result,
                    'volatility': volatility_result,
                    'momentum': momentum_result
                },
                'regime_stats': regime_stats,
                'method': 'combined'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_regime_statistics(self, features: pd.DataFrame, regimes: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for each regime"""
        try:
            stats = {}
            
            for regime in regimes.unique():
                regime_data = features[regimes == regime]
                
                if len(regime_data) == 0:
                    continue
                
                regime_stats = {
                    'count': len(regime_data),
                    'percentage': len(regime_data) / len(features) * 100,
                    'mean_returns': regime_data['returns'].mean() if 'returns' in regime_data.columns else 0,
                    'volatility': regime_data['volatility'].mean() if 'volatility' in regime_data.columns else 0,
                    'rsi': regime_data['rsi'].mean() if 'rsi' in regime_data.columns else 50,
                }
                
                stats[f'regime_{regime}'] = regime_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating regime statistics: {e}")
            return {}
    
    def create_visualizations(self, data: pd.DataFrame, regime_result: Dict[str, Any], 
                            output_dir: Path) -> List[Path]:
        """Create comprehensive regime visualizations"""
        if not self.config.generate_plots:
            return []
        
        plots = []
        
        try:
            regimes = regime_result.get('regimes', pd.Series())
            if regimes.empty:
                return plots
            
            # 1. Price with Regime Overlay
            fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
            
            # Plot price
            ax.plot(data.index, data['Close'], label='Price', alpha=0.7, linewidth=1)
            
            # Color background by regime
            colors = self.config.regime_colors[:regimes.nunique()]
            for i, regime in enumerate(regimes.unique()):
                regime_mask = regimes == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    ax.fill_between(regime_data.index, 
                                  regime_data['Close'].min() * 0.95,
                                  regime_data['Close'].max() * 1.05,
                                  alpha=0.3, color=colors[i % len(colors)],
                                  label=f'Regime {regime}')
            
            ax.set_title('Price with Regime Overlay')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            price_regime_path = output_dir / 'price_regime_overlay.png'
            plt.savefig(price_regime_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            plots.append(price_regime_path)
            
            # 2. Regime Distribution
            fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
            
            regime_counts = regimes.value_counts().sort_index()
            bars = ax.bar(regime_counts.index, regime_counts.values, 
                         color=colors[:len(regime_counts)], alpha=0.7)
            
            ax.set_title('Regime Distribution')
            ax.set_xlabel('Regime')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels
            total = len(regimes)
            for bar, count in zip(bars, regime_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{count}\n({count/total*100:.1f}%)',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            
            distribution_path = output_dir / 'regime_distribution.png'
            plt.savefig(distribution_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            plots.append(distribution_path)
            
            # 3. Regime Statistics Heatmap
            if 'regime_stats' in regime_result and regime_result['regime_stats']:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                stats_df = pd.DataFrame(regime_result['regime_stats']).T
                
                # Select numeric columns for heatmap
                numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sns.heatmap(stats_df[numeric_cols], annot=True, cmap='coolwarm', 
                               center=0, ax=ax, cbar_kws={'shrink': 0.8})
                    ax.set_title('Regime Statistics Heatmap')
                    plt.tight_layout()
                    
                    heatmap_path = output_dir / 'regime_statistics_heatmap.png'
                    plt.savefig(heatmap_path, dpi=self.config.dpi, bbox_inches='tight')
                    plt.close()
                    plots.append(heatmap_path)
            
            # 4. Volatility by Regime
            if 'volatility' in data.columns:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                for i, regime in enumerate(regimes.unique()):
                    regime_mask = regimes == regime
                    regime_vol = data['volatility'][regime_mask]
                    
                    ax.hist(regime_vol, alpha=0.6, label=f'Regime {regime}', 
                           color=colors[i % len(colors)], bins=20)
                
                ax.set_title('Volatility Distribution by Regime')
                ax.set_xlabel('Volatility')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                volatility_path = output_dir / 'volatility_by_regime.png'
                plt.savefig(volatility_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(volatility_path)
            
            # 5. Returns by Regime
            if 'returns' in data.columns:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                returns_by_regime = []
                regime_labels = []
                
                for regime in regimes.unique():
                    regime_mask = regimes == regime
                    regime_returns = data['returns'][regime_mask].dropna()
                    returns_by_regime.append(regime_returns)
                    regime_labels.append(f'Regime {regime}')
                
                ax.boxplot(returns_by_regime, labels=regime_labels)
                ax.set_title('Returns Distribution by Regime')
                ax.set_ylabel('Returns')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                returns_path = output_dir / 'returns_by_regime.png'
                plt.savefig(returns_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(returns_path)
            
            # 6. Regime Transition Matrix
            if len(regimes) > 1:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                # Create transition matrix
                transitions = pd.crosstab(regimes.shift(), regimes, dropna=False)
                transition_probs = transitions.div(transitions.sum(axis=1), axis=0)
                
                sns.heatmap(transition_probs, annot=True, cmap='Blues', 
                           ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title('Regime Transition Probabilities')
                ax.set_xlabel('Next Regime')
                ax.set_ylabel('Current Regime')
                plt.tight_layout()
                
                transition_path = output_dir / 'regime_transitions.png'
                plt.savefig(transition_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(transition_path)
            
        except Exception as e:
            self.logger.error(f"Error creating regime visualizations: {e}")
        
        return plots
    
    def run_single_backtest(self, data_file: Path, strategy_file: Path) -> List[Dict[str, Any]]:
        """Run single regime analysis backtest"""
        try:
            # Load data
            data = self.load_data(data_file)
            if data is None or data.empty:
                return []
            
            # Load strategy
            strategy_cls = self.load_strategy(strategy_file)
            if strategy_cls is None:
                return []
            
            # Create regime features
            features = self.create_regime_features(data)
            if features.empty:
                return []
            
            # Detect regimes
            regime_result = self.detect_regimes(features)
            
            if not regime_result.get('success', False):
                self.logger.warning(f"Regime detection failed for {strategy_file.name}")
                return []
            
            # Create result
            result = {
                'data_file': data_file.name,
                'strategy_file': strategy_file.name,
                'engine': 'OptimizedRegimeEngine',
                'timestamp': datetime.now().isoformat(),
                'detection_method': regime_result['method'],
                'n_regimes': regime_result['n_regimes'],
                'regime_stats': regime_result['regime_stats'],
                'regime_duration_stats': self._calculate_regime_durations(regime_result['regimes'])
            }
            
            # Add method-specific metrics
            if regime_result['method'] == 'kmeans':
                result['silhouette_score'] = regime_result.get('silhouette_score', 0)
            elif regime_result['method'] == 'gaussian_mixture':
                result['aic'] = regime_result.get('aic', 0)
                result['bic'] = regime_result.get('bic', 0)
            
            # Create visualizations
            if self.config.generate_plots:
                output_dir = self.get_results_directory() / 'visualizations' / f"{data_file.stem}_{strategy_file.stem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                plots = self.create_visualizations(data, regime_result, output_dir)
                result['visualizations'] = [str(p) for p in plots]
            
            return [result]
            
        except Exception as e:
            self.logger.error(f"Error in regime backtest: {e}")
            return []
    
    def _calculate_regime_durations(self, regimes: pd.Series) -> Dict[str, Any]:
        """Calculate regime duration statistics"""
        try:
            durations = []
            current_regime = None
            current_duration = 0
            
            for regime in regimes:
                if regime == current_regime:
                    current_duration += 1
                else:
                    if current_regime is not None:
                        durations.append(current_duration)
                    current_regime = regime
                    current_duration = 1
            
            if current_duration > 0:
                durations.append(current_duration)
            
            if durations:
                return {
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'std_duration': np.std(durations)
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calculating regime durations: {e}")
            return {}
    
    def run(self):
        """Main execution method"""
        # Discover files
        data_files = self.discover_data_files()
        strategy_files = self.discover_strategy_files()
        
        if not data_files or not strategy_files:
            self.logger.error("No data files or strategy files found")
            return
        
        # Process combinations
        all_results = []
        total_combinations = len(data_files) * len(strategy_files)
        
        self.logger.info(f"Processing {total_combinations} regime analysis combinations...")
        
        for i, data_file in enumerate(data_files):
            for j, strategy_file in enumerate(strategy_files):
                combination_num = i * len(strategy_files) + j + 1
                self.logger.info(f"Processing combination {combination_num}/{total_combinations}: "
                               f"{data_file.name} + {strategy_file.name}")
                
                try:
                    results = self.run_single_backtest(data_file, strategy_file)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error processing {data_file.name} + {strategy_file.name}: {e}")
        
        # Save results
        self.save_results(all_results)
        
        # Final summary
        if all_results:
            df = pd.DataFrame(all_results)
            self.logger.info(f"Regime Analysis Summary:")
            self.logger.info(f"   • Total combinations: {len(all_results)}")
            self.logger.info(f"   • Strategies tested: {df['strategy_file'].nunique()}")
            self.logger.info(f"   • Data files tested: {df['data_file'].nunique()}")
            self.logger.info(f"   • Average regimes detected: {df['n_regimes'].mean():.1f}")
            self.logger.info(f"   • Detection methods used: {df['detection_method'].nunique()}")
    
    def save_engine_specific_formats(self, result: Dict[str, Any], data_dir: Path):
        """Save regime-specific formats"""
        try:
            # Save regime statistics as CSV
            if 'regime_stats' in result and result['regime_stats']:
                regime_stats_data = []
                for regime_key, stats in result['regime_stats'].items():
                    regime_stats_data.append({
                        'Regime': regime_key,
                        'Count': stats.get('count', 0),
                        'Percentage': stats.get('percentage', 0),
                        'Mean Returns': stats.get('mean_returns', 0),
                        'Volatility': stats.get('volatility', 0),
                        'RSI': stats.get('rsi', 50)
                    })
                
                regime_stats_df = pd.DataFrame(regime_stats_data)
                regime_stats_path = data_dir / f"{data_dir.name}_regime_statistics.csv"
                regime_stats_df.to_csv(regime_stats_path, index=False)
            
            # Save regime duration statistics
            if 'regime_duration_stats' in result and result['regime_duration_stats']:
                duration_stats = result['regime_duration_stats']
                duration_data = {
                    'Metric': ['Mean Duration', 'Median Duration', 'Min Duration', 'Max Duration', 'Std Duration'],
                    'Value': [
                        duration_stats.get('mean_duration', 0),
                        duration_stats.get('median_duration', 0),
                        duration_stats.get('min_duration', 0),
                        duration_stats.get('max_duration', 0),
                        duration_stats.get('std_duration', 0)
                    ]
                }
                
                duration_df = pd.DataFrame(duration_data)
                duration_path = data_dir / f"{data_dir.name}_regime_durations.csv"
                duration_df.to_csv(duration_path, index=False)
            
            # Save detection method details
            detection_details = {
                'detection_method': result.get('detection_method', 'unknown'),
                'n_regimes': result.get('n_regimes', 0),
                'features_used': [fe.value for fe in self.config.features],
                'min_regime_duration': self.config.min_regime_duration,
                'regime_stability_threshold': self.config.regime_stability_threshold
            }
            
            # Add method-specific metrics
            if result.get('detection_method') == 'kmeans':
                detection_details['silhouette_score'] = result.get('silhouette_score', 0)
            elif result.get('detection_method') == 'gaussian_mixture':
                detection_details['aic'] = result.get('aic', 0)
                detection_details['bic'] = result.get('bic', 0)
            
            detection_path = data_dir / f"{data_dir.name}_detection_details.json"
            with open(detection_path, 'w') as f:
                json.dump(detection_details, f, indent=2)
            
            # Save regime summary
            regime_summary = {
                'total_regimes': result.get('n_regimes', 0),
                'detection_method': result.get('detection_method', 'unknown'),
                'regime_breakdown': result.get('regime_stats', {}),
                'duration_analysis': result.get('regime_duration_stats', {})
            }
            
            summary_path = data_dir / f"{data_dir.name}_regime_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(regime_summary, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving regime-specific formats: {e}")

def main():
    """Main entry point"""
    config = RegimeEngineConfig()
    engine = OptimizedRegimeEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
