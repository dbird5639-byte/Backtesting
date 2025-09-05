import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def calculate_market_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate market features for regime analysis.

    Mirrors the logic currently duplicated across engines so all engines
    call one implementation.
    """
    features = pd.DataFrame(index=data.index)
    returns = data['Close'].pct_change()

    # Volatility features
    features['volatility'] = returns.rolling(20, min_periods=1).std() * np.sqrt(252)
    features['volatility_ma'] = features['volatility'].rolling(10, min_periods=1).mean()

    # Trend features
    sma_20 = data['Close'].rolling(20, min_periods=1).mean()
    sma_50 = data['Close'].rolling(50, min_periods=1).mean()
    features['trend_strength'] = abs(sma_20 / sma_50 - 1)
    features['trend_direction'] = np.where(sma_20 > sma_50, 1, -1)

    # Volume features
    volume_ma = data['Volume'].rolling(20, min_periods=1).mean()
    features['volume_profile'] = data['Volume'] / volume_ma
    features['volume_trend'] = volume_ma.pct_change(5)

    # Momentum features
    features['momentum'] = data['Close'].pct_change(5)
    features['momentum_ma'] = features['momentum'].rolling(10, min_periods=1).mean()

    # Mean reversion features
    close_ma = data['Close'].rolling(20, min_periods=1).mean()
    close_std = data['Close'].rolling(20, min_periods=1).std()
    bb_std = (data['Close'] - close_ma) / close_std
    features['mean_reversion'] = bb_std
    features['mean_reversion_ma'] = bb_std.rolling(10, min_periods=1).mean()

    # Price action features
    features['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
    features['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])

    # Clean features
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

    return features


def detect_regimes(features: pd.DataFrame, n_regimes: int = 3) -> Tuple[np.ndarray, List[Dict]]:
    """Detect market regimes using clustering and summarize each cluster."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        cluster_features = [
            'volatility_ma', 'trend_strength', 'volume_profile',
            'momentum_ma', 'mean_reversion_ma'
        ]

        X = features[cluster_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        regimes: List[Dict] = []
        for i in range(n_regimes):
            cluster_mask = clusters == i
            cluster_features_subset = features[cluster_mask]
            regime = {
                'name': f"Regime_{i+1}",
                'volatility': float(cluster_features_subset['volatility_ma'].mean()),
                'trend_strength': float(cluster_features_subset['trend_strength'].mean()),
                'volume_profile': float(cluster_features_subset['volume_profile'].mean()),
                'momentum': float(cluster_features_subset['momentum_ma'].mean()),
                'mean_reversion': float(cluster_features_subset['mean_reversion_ma'].mean()),
                'count': int(len(cluster_features_subset))
            }
            regimes.append(regime)

        return clusters, regimes
    except Exception:
        return np.zeros(len(features)), []


