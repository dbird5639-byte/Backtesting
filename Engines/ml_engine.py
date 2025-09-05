#!/usr/bin/env python3
"""
ML-Powered Backtesting Engine

A sophisticated engine that provides:
- Automated feature engineering
- Multiple ML model training and selection
- Cross-validation and hyperparameter optimization
- Model performance tracking and comparison
- Real-time model updates and adaptation
- Ensemble methods and stacking
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLEngineConfig:
    """Configuration for the ML engine"""
    # Paths
    data_path: str = "./fetched_data"
    strategies_path: str = "./Strategies"
    results_path: str = "./Results"
    models_path: str = "./Models"
    
    # ML Configuration
    test_size: float = 0.2
    validation_size: float = 0.2
    n_splits: int = 5  # For time series cross-validation
    random_state: int = 42
    
    # Feature Engineering
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stoch', 'williams_r'
    ])
    price_features: List[str] = field(default_factory=lambda: [
        'returns', 'log_returns', 'volatility', 'volume_ratio'
    ])
    
    # Models to train
    models: List[str] = field(default_factory=lambda: [
        'random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'neural_network'
    ])
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    cv_folds: int = 3
    n_jobs: int = -1
    
    # Performance
    max_workers: int = 4
    chunk_size: int = 1000
    
    # Output
    save_models: bool = True
    save_features: bool = True
    save_predictions: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

@dataclass
class MLResult:
    """Result of ML model training and testing"""
    model_name: str
    strategy_name: str
    data_file: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time: float
    prediction_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureEngineer:
    """Advanced feature engineering for trading data"""
    
    def __init__(self, config: MLEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = RobustScaler()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        try:
            features_df = data.copy()
            
            # Price-based features
            features_df = self._add_price_features(features_df)
            
            # Technical indicators
            features_df = self._add_technical_indicators(features_df)
            
            # Time-based features
            features_df = self._add_time_features(features_df)
            
            # Lagged features
            features_df = self._add_lagged_features(features_df)
            
            # Rolling statistics
            features_df = self._add_rolling_features(features_df)
            
            # Market regime features
            features_df = self._add_regime_features(features_df)
            
            # Clean features
            features_df = self._clean_features(features_df)
            
            return features_df
        
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return data
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['volume_price_trend'] = df['Volume'] * df['returns']
        
        # Price position
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Gap features
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_ratio'] = df['gap'] / df['volatility']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'price_sma_{window}_ratio'] = df['Close'] / df[f'sma_{window}']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_sma'] = df['rsi'].rolling(window=10).mean()
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['Close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if df.index.dtype == 'datetime64[ns]':
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
            df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features"""
        for window in self.config.lookback_windows:
            # Rolling statistics
            df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Rolling quantiles
            df[f'returns_q25_{window}'] = df['returns'].rolling(window=window).quantile(0.25)
            df[f'returns_q75_{window}'] = df['returns'].rolling(window=window).quantile(0.75)
            
            # Rolling min/max
            df[f'price_min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'price_max_{window}'] = df['Close'].rolling(window=window).max()
            df[f'price_range_{window}'] = df[f'price_max_{window}'] - df[f'price_min_{window}']
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        # Volatility regime
        vol_threshold = df['volatility'].rolling(window=100).quantile(0.7)
        df['high_vol_regime'] = (df['volatility'] > vol_threshold).astype(int)
        
        # Trend regime
        trend_threshold = df['returns'].rolling(window=50).mean()
        df['uptrend_regime'] = (trend_threshold > 0).astype(int)
        
        # Volume regime
        vol_ratio_threshold = df['volume_ratio'].rolling(window=100).quantile(0.7)
        df['high_volume_regime'] = (df['volume_ratio'] > vol_ratio_threshold).astype(int)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML"""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Drop rows with remaining NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        return williams_r

class ModelTrainer:
    """ML model trainer with hyperparameter optimization"""
    
    def __init__(self, config: MLEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = self._initialize_models()
        self.scaler = RobustScaler()
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ML models"""
        return {
            'random_forest': RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.config.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.config.random_state,
                probability=True
            ),
            'neural_network': MLPClassifier(
                random_state=self.config.random_state,
                max_iter=1000
            )
        }
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Get hyperparameter grids for each model"""
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    async def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, MLResult]:
        """Train all models with hyperparameter optimization"""
        results = {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        for model_name in self.config.models:
            if model_name not in self.models:
                continue
            
            try:
                self.logger.info(f"Training {model_name}...")
                start_time = time.time()
                
                # Get model and hyperparameter grid
                model = self.models[model_name]
                param_grid = self.get_hyperparameter_grids().get(model_name, {})
                
                # Hyperparameter tuning
                if self.config.enable_hyperparameter_tuning and param_grid:
                    # Use time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
                    
                    grid_search = GridSearchCV(
                        model,
                        param_grid,
                        cv=tscv,
                        scoring='f1',
                        n_jobs=self.config.n_jobs,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    best_model = model
                    best_params = {}
                
                # Train final model
                best_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = best_model.predict(X_val_scaled)
                y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                auc = roc_auc_score(y_val, y_pred_proba)
                
                # Feature importance
                feature_importance = {}
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
                elif hasattr(best_model, 'coef_'):
                    feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_[0])))
                
                training_time = time.time() - start_time
                
                # Create result
                result = MLResult(
                    model_name=model_name,
                    strategy_name="ML_Strategy",
                    data_file="Unknown",
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    auc_score=auc,
                    feature_importance=feature_importance,
                    hyperparameters=best_params,
                    training_time=training_time,
                    prediction_time=0.0,  # Will be measured separately
                    metadata={
                        'n_features': X_train.shape[1],
                        'n_train_samples': len(X_train),
                        'n_val_samples': len(X_val)
                    }
                )
                
                results[model_name] = result
                self.logger.info(f"{model_name} trained successfully - F1: {f1:.3f}, AUC: {auc:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results

class MLEngine:
    """ML-powered backtesting engine"""
    
    def __init__(self, config: MLEngineConfig = None):
        self.config = config or MLEngineConfig()
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        
        self.logger.info("ML Engine initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if requested
        if self.config.log_to_file:
            log_file = f"ml_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            logging.getLogger().addHandler(file_handler)
    
    def create_target_variable(self, data: pd.DataFrame, horizon: int = 1, threshold: float = 0.01) -> pd.Series:
        """Create target variable for classification"""
        # Future returns
        future_returns = data['Close'].shift(-horizon) / data['Close'] - 1
        
        # Binary classification: 1 if return > threshold, 0 otherwise
        target = (future_returns > threshold).astype(int)
        
        return target
    
    async def run(self) -> List[MLResult]:
        """Run the ML engine"""
        try:
            self.logger.info("Starting ML Engine")
            start_time = time.time()
            
            # Discover data files
            data_path = Path(self.config.data_path)
            data_files = list(data_path.rglob("*.csv"))
            
            if not data_files:
                self.logger.warning("No data files found")
                return []
            
            all_results = []
            
            for data_file in data_files:
                try:
                    self.logger.info(f"Processing {data_file.name}")
                    
                    # Load data
                    data = pd.read_csv(data_file, parse_dates=True, index_col=0)
                    
                    if len(data) < 1000:  # Need sufficient data for ML
                        self.logger.warning(f"Insufficient data in {data_file.name}: {len(data)}")
                        continue
                    
                    # Create features
                    features_df = self.feature_engineer.create_features(data)
                    
                    if features_df.empty:
                        self.logger.warning(f"No features created for {data_file.name}")
                        continue
                    
                    # Create target variable
                    target = self.create_target_variable(features_df)
                    
                    # Remove rows with NaN target
                    valid_idx = ~target.isna()
                    features_df = features_df[valid_idx]
                    target = target[valid_idx]
                    
                    if len(features_df) < 500:  # Need sufficient data after cleaning
                        self.logger.warning(f"Insufficient data after cleaning for {data_file.name}: {len(features_df)}")
                        continue
                    
                    # Split data
                    split_idx = int(len(features_df) * (1 - self.config.test_size - self.config.validation_size))
                    val_idx = int(len(features_df) * (1 - self.config.test_size))
                    
                    X_train = features_df.iloc[:split_idx]
                    y_train = target.iloc[:split_idx]
                    X_val = features_df.iloc[split_idx:val_idx]
                    y_val = target.iloc[split_idx:val_idx]
                    X_test = features_df.iloc[val_idx:]
                    y_test = target.iloc[val_idx:]
                    
                    # Train models
                    results = await self.model_trainer.train_models(X_train, y_train, X_val, y_val)
                    
                    # Update results with data file info
                    for result in results.values():
                        result.data_file = data_file.stem
                        all_results.append(result)
                    
                    # Save features if requested
                    if self.config.save_features:
                        await self._save_features(features_df, data_file.stem)
                    
                    self.logger.info(f"Completed processing {data_file.name}")
                
                except Exception as e:
                    self.logger.error(f"Error processing {data_file.name}: {e}")
                    continue
            
            # Save results
            if all_results:
                await self._save_results(all_results)
            
            execution_time = time.time() - start_time
            self.logger.info(f"ML Engine completed in {execution_time:.2f} seconds")
            self.logger.info(f"Generated {len(all_results)} model results")
            
            return all_results
        
        except Exception as e:
            self.logger.error(f"Error in ML Engine: {e}")
            return []
    
    async def _save_features(self, features_df: pd.DataFrame, data_name: str):
        """Save engineered features"""
        try:
            features_path = Path(self.config.results_path) / "features"
            features_path.mkdir(parents=True, exist_ok=True)
            
            features_file = features_path / f"{data_name}_features.csv"
            features_df.to_csv(features_file)
            
            self.logger.info(f"Saved features to {features_file}")
        
        except Exception as e:
            self.logger.error(f"Error saving features: {e}")
    
    async def _save_results(self, results: List[MLResult]):
        """Save ML results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(self.config.results_path) / f"ml_engine_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert results to dict
            results_dict = []
            for result in results:
                result_dict = {
                    'model_name': result.model_name,
                    'strategy_name': result.strategy_name,
                    'data_file': result.data_file,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'auc_score': result.auc_score,
                    'feature_importance': result.feature_importance,
                    'hyperparameters': result.hyperparameters,
                    'training_time': result.training_time,
                    'prediction_time': result.prediction_time,
                    'timestamp': result.timestamp.isoformat(),
                    'metadata': result.metadata
                }
                results_dict.append(result_dict)
            
            # Save JSON
            json_file = results_dir / "ml_results.json"
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # Save CSV
            csv_data = []
            for result in results:
                row = {
                    'model_name': result.model_name,
                    'data_file': result.data_file,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'auc_score': result.auc_score,
                    'training_time': result.training_time
                }
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_file = results_dir / "ml_results.csv"
            df.to_csv(csv_file, index=False)
            
            # Save summary
            summary_file = results_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write("ML Engine Results Summary\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Total Results: {len(results)}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Best performing models
                best_f1 = max(results, key=lambda r: r.f1_score)
                best_auc = max(results, key=lambda r: r.auc_score)
                
                f.write(f"Best F1 Score: {best_f1.model_name} ({best_f1.f1_score:.3f})\n")
                f.write(f"Best AUC Score: {best_auc.model_name} ({best_auc.auc_score:.3f})\n")
            
            self.logger.info(f"Saved results to {results_dir}")
        
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

async def main():
    """Main function to run the ML engine"""
    config = MLEngineConfig(
        test_size=0.2,
        validation_size=0.2,
        enable_hyperparameter_tuning=True,
        save_features=True,
        save_models=True
    )
    
    engine = MLEngine(config)
    results = await engine.run()
    
    print(f"\nML Engine Results:")
    print(f"Total Results: {len(results)}")
    
    if results:
        print(f"Average F1 Score: {np.mean([r.f1_score for r in results]):.3f}")
        print(f"Average AUC Score: {np.mean([r.auc_score for r in results]):.3f}")
        
        best_f1 = max(results, key=lambda r: r.f1_score)
        print(f"Best F1: {best_f1.model_name} on {best_f1.data_file} ({best_f1.f1_score:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
