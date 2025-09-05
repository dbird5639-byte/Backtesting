#!/usr/bin/env python3
"""
Historical Regime Analysis Script

This script analyzes historical market data to identify and classify different market regimes
across various tokens and timeframes. It provides comprehensive regime detection and reporting
for use in backtesting and bot decision-making.

Features:
- Multiple regime detection algorithms (volatility, trend, momentum, volume)
- Support for multiple timeframes and tokens
- Comprehensive regime classification
- Historical regime reporting with statistics
- Export to CSV and JSON for integration with backtesting
- Regime transition analysis
- Regime persistence and stability metrics
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import required libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Some visualization libraries not available. Install matplotlib, seaborn, plotly for full functionality.")

@dataclass
class RegimeConfig:
    """Configuration for regime analysis"""
    # Paths
    data_path: str = "./Data"
    results_path: str = "./Results/RegimeAnalysis"
    
    # Regime detection parameters
    lookback_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    volatility_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.1, 'medium': 0.2, 'high': 0.3, 'extreme': 0.5
    })
    trend_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'strong_bull': 0.05, 'bull': 0.02, 'sideways': 0.01, 'bear': -0.02, 'strong_bear': -0.05
    })
    
    # Clustering parameters
    n_regimes: int = 4
    min_regime_duration: int = 5  # Minimum periods for a regime
    regime_stability_threshold: float = 0.7
    
    # Output options
    save_csv: bool = True
    save_json: bool = True
    save_plots: bool = True
    save_heatmaps: bool = True
    
    # Analysis options
    include_volume_analysis: bool = True
    include_correlation_analysis: bool = True
    include_transition_analysis: bool = True

@dataclass
class RegimeResult:
    """Result of regime analysis for a single data file"""
    data_file: str
    timeframe: str
    token: str
    total_periods: int
    regimes_detected: int
    regime_breakdown: Dict[str, int]
    regime_statistics: Dict[str, Any]
    regime_transitions: List[Dict[str, Any]]
    regime_persistence: Dict[str, float]
    regime_stability: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class RegimeDetector:
    """Advanced regime detection using multiple algorithms"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def detect_volatility_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect volatility-based regimes"""
        returns = data['close'].pct_change().dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=20).std()
        
        # Classify volatility regimes
        regimes = pd.Series(index=volatility.index, dtype='object')
        regimes[volatility <= self.config.volatility_thresholds['low']] = 'Low_Volatility'
        regimes[(volatility > self.config.volatility_thresholds['low']) & 
                (volatility <= self.config.volatility_thresholds['medium'])] = 'Medium_Volatility'
        regimes[(volatility > self.config.volatility_thresholds['medium']) & 
                (volatility <= self.config.volatility_thresholds['high'])] = 'High_Volatility'
        regimes[volatility > self.config.volatility_thresholds['extreme']] = 'Extreme_Volatility'
        
        return regimes
    
    def detect_trend_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect trend-based regimes"""
        close = data['close']
        
        # Calculate trend strength using multiple lookback periods
        trend_signals = []
        for period in self.config.lookback_periods:
            sma = close.rolling(window=period).mean()
            trend_strength = (close - sma) / sma
            trend_signals.append(trend_strength)
        
        # Combine trend signals
        combined_trend = pd.concat(trend_signals, axis=1).mean(axis=1)
        
        # Classify trend regimes
        regimes = pd.Series(index=combined_trend.index, dtype='object')
        regimes[combined_trend >= self.config.trend_thresholds['strong_bull']] = 'Strong_Bull'
        regimes[(combined_trend >= self.config.trend_thresholds['bull']) & 
                (combined_trend < self.config.trend_thresholds['strong_bull'])] = 'Bull'
        regimes[(combined_trend >= -self.config.trend_thresholds['sideways']) & 
                (combined_trend < self.config.trend_thresholds['bull'])] = 'Sideways'
        regimes[(combined_trend >= self.config.trend_thresholds['bear']) & 
                (combined_trend < -self.config.trend_thresholds['sideways'])] = 'Bear'
        regimes[combined_trend < self.config.trend_thresholds['strong_bear']] = 'Strong_Bear'
        
        return regimes
    
    def detect_momentum_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect momentum-based regimes"""
        close = data['close']
        
        # Calculate momentum indicators
        rsi = self.calculate_rsi(close, 14)
        macd_line, macd_signal = self.calculate_macd(close)
        momentum = close.pct_change(10)
        
        # Combine momentum signals
        momentum_score = (rsi - 50) / 50 + (macd_line - macd_signal) / macd_signal + momentum
        
        # Classify momentum regimes
        regimes = pd.Series(index=momentum_score.index, dtype='object')
        regimes[momentum_score > 0.3] = 'Strong_Momentum'
        regimes[(momentum_score > 0.1) & (momentum_score <= 0.3)] = 'Momentum'
        regimes[(momentum_score >= -0.1) & (momentum_score <= 0.1)] = 'Neutral'
        regimes[(momentum_score >= -0.3) & (momentum_score < -0.1)] = 'Negative_Momentum'
        regimes[momentum_score < -0.3] = 'Strong_Negative_Momentum'
        
        return regimes
    
    def detect_volume_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect volume-based regimes"""
        if 'volume' not in data.columns:
            return pd.Series(index=data.index, dtype='object')
        
        volume = data['volume']
        volume_sma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_sma
        
        # Classify volume regimes
        regimes = pd.Series(index=volume_ratio.index, dtype='object')
        regimes[volume_ratio > 2.0] = 'Extreme_Volume'
        regimes[(volume_ratio > 1.5) & (volume_ratio <= 2.0)] = 'High_Volume'
        regimes[(volume_ratio > 0.8) & (volume_ratio <= 1.5)] = 'Normal_Volume'
        regimes[(volume_ratio > 0.5) & (volume_ratio <= 0.8)] = 'Low_Volume'
        regimes[volume_ratio <= 0.5] = 'Very_Low_Volume'
        
        return regimes
    
    def detect_cluster_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes using clustering analysis"""
        # Prepare features for clustering
        features = []
        
        # Price features
        close = data['close']
        returns = close.pct_change().dropna()
        features.append(returns.rolling(20).mean())  # Mean return
        features.append(returns.rolling(20).std())   # Volatility
        features.append(returns.rolling(20).skew())  # Skewness
        features.append(returns.rolling(20).kurt())  # Kurtosis
        
        # Technical indicators
        features.append(self.calculate_rsi(close, 14))
        macd_line, _ = self.calculate_macd(close)
        features.append(macd_line)
        
        # Volume features (if available)
        if 'volume' in data.columns:
            volume = data['volume']
            features.append(volume.rolling(20).mean())
            features.append(volume.pct_change().rolling(20).std())
        
        # Combine features
        feature_df = pd.concat(features, axis=1).dropna()
        
        if len(feature_df) < self.config.n_regimes * 2:
            return pd.Series(index=data.index, dtype='object')
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Create regime series
        regimes = pd.Series(index=feature_df.index, dtype='object')
        for i, label in enumerate(cluster_labels):
            regimes.iloc[i] = f'Cluster_{label}'
        
        return regimes
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line, macd_signal
    
    def analyze_regime_transitions(self, regimes: pd.Series) -> List[Dict[str, Any]]:
        """Analyze regime transitions"""
        transitions = []
        regime_changes = regimes != regimes.shift(1)
        transition_points = regime_changes[regime_changes].index
        
        for i, point in enumerate(transition_points):
            if i == 0:
                continue
            
            prev_regime = regimes.iloc[regime_changes.index.get_loc(point) - 1]
            current_regime = regimes.iloc[regime_changes.index.get_loc(point)]
            
            # Calculate regime duration
            if i > 0:
                prev_transition = transition_points[i-1]
                duration = (point - prev_transition).days if hasattr(point, 'days') else 1
            else:
                duration = 1
            
            transitions.append({
                'timestamp': point,
                'from_regime': prev_regime,
                'to_regime': current_regime,
                'duration_days': duration,
                'transition_type': f"{prev_regime}_to_{current_regime}"
            })
        
        return transitions
    
    def calculate_regime_statistics(self, data: pd.DataFrame, regimes: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive regime statistics"""
        stats = {}
        returns = data['close'].pct_change().dropna()
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) == 0:
                continue
            
            stats[regime] = {
                'count': len(regime_returns),
                'percentage': len(regime_returns) / len(returns) * 100,
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'min_return': regime_returns.min(),
                'max_return': regime_returns.max(),
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'positive_days': (regime_returns > 0).sum(),
                'negative_days': (regime_returns < 0).sum(),
                'win_rate': (regime_returns > 0).sum() / len(regime_returns) * 100
            }
        
        return stats
    
    def calculate_regime_persistence(self, regimes: pd.Series) -> Dict[str, float]:
        """Calculate regime persistence metrics"""
        persistence = {}
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            
            regime_mask = regimes == regime
            regime_periods = regime_mask[regime_mask]
            
            if len(regime_periods) == 0:
                continue
            
            # Calculate average duration
            durations = []
            current_duration = 0
            
            for is_regime in regime_mask:
                if is_regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            persistence[regime] = {
                'avg_duration': np.mean(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'stability': len(durations) / len(regime_mask) if len(regime_mask) > 0 else 0
            }
        
        return persistence

class RegimeAnalyzer:
    """Main regime analysis orchestrator"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.detector = RegimeDetector(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def discover_data_files(self) -> List[str]:
        """Discover all data files"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            self.logger.warning(f"Data path does not exist: {data_path}")
            return []
        
        csv_files = list(data_path.rglob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files")
        return [str(f) for f in csv_files]
    
    def extract_file_info(self, file_path: str) -> Dict[str, str]:
        """Extract token and timeframe information from filename"""
        filename = Path(file_path).stem
        
        # Common patterns for token/timeframe extraction
        # Adjust these patterns based on your file naming convention
        parts = filename.split('_')
        
        if len(parts) >= 2:
            token = parts[0].upper()
            timeframe = parts[1].upper()
        else:
            token = filename.upper()
            timeframe = "UNKNOWN"
        
        return {
            'token': token,
            'timeframe': timeframe,
            'filename': filename
        }
    
    def load_and_prepare_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and prepare data for analysis"""
        try:
            data = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'time': 'timestamp',
                'date': 'timestamp',
                'datetime': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vol': 'volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Ensure timestamp is datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                self.logger.warning(f"Missing required columns in {file_path}")
                return None
            
            # Remove rows with missing data
            data = data.dropna(subset=required_columns)
            
            if len(data) < 100:
                self.logger.warning(f"Insufficient data in {file_path}: {len(data)} rows")
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return None
    
    def analyze_single_file(self, file_path: str) -> Optional[RegimeResult]:
        """Analyze regimes for a single data file"""
        self.logger.info(f"Analyzing regimes for: {file_path}")
        
        # Load data
        data = self.load_and_prepare_data(file_path)
        if data is None:
            return None
        
        # Extract file information
        file_info = self.extract_file_info(file_path)
        
        # Detect different types of regimes
        volatility_regimes = self.detector.detect_volatility_regimes(data)
        trend_regimes = self.detector.detect_trend_regimes(data)
        momentum_regimes = self.detector.detect_momentum_regimes(data)
        
        # Combine regimes (you can modify this logic)
        combined_regimes = self.combine_regimes(volatility_regimes, trend_regimes, momentum_regimes)
        
        # Calculate statistics
        regime_stats = self.detector.calculate_regime_statistics(data, combined_regimes)
        regime_transitions = self.detector.analyze_regime_transitions(combined_regimes)
        regime_persistence = self.detector.calculate_regime_persistence(combined_regimes)
        
        # Calculate regime breakdown
        regime_breakdown = combined_regimes.value_counts().to_dict()
        
        # Calculate overall stability
        regime_stability = self.calculate_overall_stability(combined_regimes)
        
        # Create result
        result = RegimeResult(
            data_file=file_path,
            timeframe=file_info['timeframe'],
            token=file_info['token'],
            total_periods=len(data),
            regimes_detected=len(combined_regimes.unique()),
            regime_breakdown=regime_breakdown,
            regime_statistics=regime_stats,
            regime_transitions=regime_transitions,
            regime_persistence=regime_persistence,
            regime_stability=regime_stability
        )
        
        return result
    
    def combine_regimes(self, vol_regimes: pd.Series, trend_regimes: pd.Series, 
                       momentum_regimes: pd.Series) -> pd.Series:
        """Combine different regime types into a unified regime classification"""
        # Simple combination logic - you can make this more sophisticated
        combined = pd.Series(index=vol_regimes.index, dtype='object')
        
        for i in vol_regimes.index:
            vol = vol_regimes.get(i, 'Unknown')
            trend = trend_regimes.get(i, 'Unknown')
            momentum = momentum_regimes.get(i, 'Unknown')
            
            # Create combined regime name
            combined_regime = f"{vol}_{trend}_{momentum}"
            combined[i] = combined_regime
        
        return combined
    
    def calculate_overall_stability(self, regimes: pd.Series) -> float:
        """Calculate overall regime stability"""
        if len(regimes) == 0:
            return 0.0
        
        # Count regime changes
        regime_changes = (regimes != regimes.shift(1)).sum()
        stability = 1.0 - (regime_changes / len(regimes))
        
        return max(0.0, min(1.0, stability))
    
    def save_results(self, results: List[RegimeResult]) -> str:
        """Save regime analysis results"""
        os.makedirs(self.config.results_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV summary
        if self.config.save_csv:
            summary_data = []
            for result in results:
                summary_data.append({
                    'data_file': result.data_file,
                    'token': result.token,
                    'timeframe': result.timeframe,
                    'total_periods': result.total_periods,
                    'regimes_detected': result.regimes_detected,
                    'regime_stability': result.regime_stability,
                    'analysis_timestamp': result.analysis_timestamp
                })
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = f"{self.config.results_path}/regime_analysis_summary_{timestamp}.csv"
            summary_df.to_csv(csv_path, index=False)
            self.logger.info(f"Summary saved to: {csv_path}")
        
        # Save detailed JSON results
        if self.config.save_json:
            json_data = []
            for result in results:
                json_data.append({
                    'data_file': result.data_file,
                    'token': result.token,
                    'timeframe': result.timeframe,
                    'total_periods': result.total_periods,
                    'regimes_detected': result.regimes_detected,
                    'regime_breakdown': result.regime_breakdown,
                    'regime_statistics': result.regime_statistics,
                    'regime_transitions': result.regime_transitions,
                    'regime_persistence': result.regime_persistence,
                    'regime_stability': result.regime_stability,
                    'analysis_timestamp': result.analysis_timestamp.isoformat()
                })
            
            json_path = f"{self.config.results_path}/regime_analysis_detailed_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            self.logger.info(f"Detailed results saved to: {json_path}")
        
        return self.config.results_path
    
    def run_analysis(self) -> List[RegimeResult]:
        """Run complete regime analysis"""
        self.logger.info("Starting regime analysis...")
        
        # Discover data files
        data_files = self.discover_data_files()
        if not data_files:
            self.logger.error("No data files found")
            return []
        
        # Analyze each file
        results = []
        for file_path in data_files:
            result = self.analyze_single_file(file_path)
            if result:
                results.append(result)
        
        self.logger.info(f"Analysis complete. Processed {len(results)} files")
        
        # Save results
        if results:
            self.save_results(results)
        
        return results

def main():
    """Main function"""
    print("🔍 Historical Regime Analysis")
    print("=" * 50)
    
    # Create configuration
    config = RegimeConfig(
        data_path="./Data",
        results_path="./Results/RegimeAnalysis",
        save_csv=True,
        save_json=True,
        save_plots=True,
        save_heatmaps=True
    )
    
    # Create analyzer
    analyzer = RegimeAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        print(f"\n✅ Analysis complete! Processed {len(results)} files")
        print(f"📊 Results saved to: {config.results_path}")
        
        # Print summary
        print("\n📋 Analysis Summary:")
        for result in results[:5]:  # Show first 5
            print(f"  {result.token} ({result.timeframe}): {result.regimes_detected} regimes, "
                  f"stability: {result.regime_stability:.3f}")
        
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more files")
    else:
        print("❌ No results generated. Check your data files and configuration.")

if __name__ == "__main__":
    main()
