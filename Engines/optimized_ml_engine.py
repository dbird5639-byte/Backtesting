#!/usr/bin/env python3
"""
Optimized ML Engine with Integrated Visualization
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

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.feature_selection import SelectKBest, f_regression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .base_optimized_engine import BaseOptimizedEngine, BaseEngineConfig

class MLModel(Enum):
    """ML Model enumeration"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    SVR = "svr"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

class FeatureEngineering(Enum):
    """Feature engineering methods"""
    TECHNICAL_INDICATORS = "technical_indicators"
    LAGGED_FEATURES = "lagged_features"
    ROLLING_STATISTICS = "rolling_statistics"
    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"

@dataclass
class MLEngineConfig(BaseEngineConfig):
    """Configuration for ML Engine"""
    # Model settings
    model_type: MLModel = MLModel.RANDOM_FOREST
    feature_engineering: List[FeatureEngineering] = field(default_factory=lambda: [
        FeatureEngineering.TECHNICAL_INDICATORS,
        FeatureEngineering.LAGGED_FEATURES,
        FeatureEngineering.ROLLING_STATISTICS
    ])
    
    # Training settings
    test_size: float = 0.2
    validation_size: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # Feature selection
    max_features: int = 50
    feature_selection_method: str = "f_regression"
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    regularization: float = 0.01
    
    # Prediction settings
    prediction_horizon: int = 1  # days ahead
    confidence_interval: float = 0.95
    
    # Visualization settings
    generate_plots: bool = True
    plot_style: str = "seaborn-v0_8"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300

class OptimizedMLEngine(BaseOptimizedEngine):
    """Optimized ML Engine with integrated visualization"""
    
    def __init__(self, config: MLEngineConfig = None):
        super().__init__(config or MLEngineConfig())
        self.config: MLEngineConfig = self.config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available. Install scikit-learn for ML functionality.")
        
        # Setup visualization
        if self.config.generate_plots:
            plt.style.use(self.config.plot_style)
            sns.set_palette("husl")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        try:
            features = data.copy()
            
            # Technical indicators
            if FeatureEngineering.TECHNICAL_INDICATORS in self.config.feature_engineering:
                features = self._add_technical_indicators(features)
            
            # Lagged features
            if FeatureEngineering.LAGGED_FEATURES in self.config.feature_engineering:
                features = self._add_lagged_features(features)
            
            # Rolling statistics
            if FeatureEngineering.ROLLING_STATISTICS in self.config.feature_engineering:
                features = self._add_rolling_statistics(features)
            
            # Polynomial features
            if FeatureEngineering.POLYNOMIAL in self.config.feature_engineering:
                features = self._add_polynomial_features(features)
            
            # Interaction features
            if FeatureEngineering.INTERACTION in self.config.feature_engineering:
                features = self._add_interaction_features(features)
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
            df = data.copy()
            
            # Price-based indicators
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['sma_50'] = df['Close'].rolling(50).mean()
            
            df['ema_5'] = df['Close'].ewm(span=5).mean()
            df['ema_20'] = df['Close'].ewm(span=20).mean()
            
            # Volatility indicators
            df['bb_upper'] = df['sma_20'] + (df['Close'].rolling(20).std() * 2)
            df['bb_lower'] = df['sma_20'] - (df['Close'].rolling(20).std() * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
            
            # Momentum indicators
            df['rsi'] = self._calculate_rsi(df['Close'])
            df['macd'] = df['ema_12'] - df['ema_26'] if 'ema_12' in df.columns else 0
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Volume indicators
            if 'Volume' in df.columns:
                df['volume_sma'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
                df['price_volume'] = df['Close'] * df['Volume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _add_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        try:
            df = data.copy()
            
            # Price lags
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
            
            # Return lags
            returns = df['Close'].pct_change()
            for lag in [1, 2, 3, 5]:
                df[f'return_lag_{lag}'] = returns.shift(lag)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding lagged features: {e}")
            return data
    
    def _add_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics"""
        try:
            df = data.copy()
            
            for window in [5, 10, 20, 50]:
                df[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
                df[f'close_std_{window}'] = df['Close'].rolling(window).std()
                df[f'close_min_{window}'] = df['Close'].rolling(window).min()
                df[f'close_max_{window}'] = df['Close'].rolling(window).max()
                df[f'close_skew_{window}'] = df['Close'].rolling(window).skew()
                df[f'close_kurt_{window}'] = df['Close'].rolling(window).kurt()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling statistics: {e}")
            return data
    
    def _add_polynomial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features"""
        try:
            df = data.copy()
            
            # Price polynomials
            df['close_squared'] = df['Close'] ** 2
            df['close_cubed'] = df['Close'] ** 3
            
            # Volume polynomials
            if 'Volume' in df.columns:
                df['volume_squared'] = df['Volume'] ** 2
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding polynomial features: {e}")
            return data
    
    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        try:
            df = data.copy()
            
            # Price-volume interactions
            if 'Volume' in df.columns:
                df['price_volume'] = df['Close'] * df['Volume']
                df['price_volume_ratio'] = df['Close'] / df['Volume']
            
            # Technical indicator interactions
            if 'sma_5' in df.columns and 'sma_20' in df.columns:
                df['sma_ratio'] = df['sma_5'] / df['sma_20']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding interaction features: {e}")
            return data
    
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
    
    def create_model(self, model_type: MLModel = None) -> Any:
        """Create ML model"""
        if not ML_AVAILABLE:
            return None
        
        model_type = model_type or self.config.model_type
        
        try:
            if model_type == MLModel.RANDOM_FOREST:
                return RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state
                )
            elif model_type == MLModel.GRADIENT_BOOSTING:
                return GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state
                )
            elif model_type == MLModel.LINEAR_REGRESSION:
                return LinearRegression()
            elif model_type == MLModel.RIDGE:
                return Ridge(alpha=self.config.regularization)
            elif model_type == MLModel.LASSO:
                return Lasso(alpha=self.config.regularization)
            elif model_type == MLModel.SVR:
                return SVR(kernel='rbf', C=1.0, gamma='scale')
            elif model_type == MLModel.NEURAL_NETWORK:
                return MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=self.config.random_state
                )
            else:
                return RandomForestRegressor(random_state=self.config.random_state)
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            return None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: MLModel = None) -> Dict[str, Any]:
        """Train ML model"""
        if not ML_AVAILABLE:
            return {'success': False, 'error': 'ML libraries not available'}
        
        try:
            # Create model
            model = self.create_model(model_type)
            if model is None:
                return {'success': False, 'error': 'Failed to create model'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=self.config.cross_validation_folds, scoring='r2')
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            return {
                'success': True,
                'model': model,
                'scaler': scaler,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred_test': y_pred_test
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_visualizations(self, data: pd.DataFrame, results: Dict[str, Any], 
                            output_dir: Path) -> List[Path]:
        """Create comprehensive visualizations"""
        if not self.config.generate_plots or not ML_AVAILABLE:
            return []
        
        plots = []
        
        try:
            # 1. Feature Importance Plot
            if 'feature_importance' in results and results['feature_importance']:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                importance = results['feature_importance']
                features = list(importance.keys())
                values = list(importance.values())
                
                # Sort by importance
                sorted_idx = np.argsort(values)[::-1][:20]  # Top 20 features
                
                ax.barh(range(len(sorted_idx)), [values[i] for i in sorted_idx])
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([features[i] for i in sorted_idx])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 20 Feature Importance')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                importance_path = output_dir / 'feature_importance.png'
                plt.savefig(importance_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(importance_path)
            
            # 2. Prediction vs Actual Plot
            if 'y_test' in results and 'y_pred_test' in results:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                y_test = results['y_test']
                y_pred = results['y_pred_test']
                
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Prediction vs Actual')
                ax.grid(True, alpha=0.3)
                
                # Add R² score
                r2 = results.get('test_r2', 0)
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                
                pred_actual_path = output_dir / 'prediction_vs_actual.png'
                plt.savefig(pred_actual_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(pred_actual_path)
            
            # 3. Residuals Plot
            if 'y_test' in results and 'y_pred_test' in results:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                y_test = results['y_test']
                y_pred = results['y_pred_test']
                residuals = y_test - y_pred
                
                ax.scatter(y_pred, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residuals Plot')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                residuals_path = output_dir / 'residuals_plot.png'
                plt.savefig(residuals_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(residuals_path)
            
            # 4. Model Performance Comparison
            if 'cv_mean' in results and 'cv_std' in results:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                metrics = ['Train R²', 'Test R²', 'CV Mean']
                values = [results.get('train_r2', 0), results.get('test_r2', 0), results.get('cv_mean', 0)]
                errors = [0, 0, results.get('cv_std', 0)]
                
                bars = ax.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7)
                ax.set_ylabel('R² Score')
                ax.set_title('Model Performance Metrics')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                performance_path = output_dir / 'model_performance.png'
                plt.savefig(performance_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(performance_path)
            
            # 5. Time Series Prediction
            if len(data) > 0:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                # Plot actual prices
                ax.plot(data.index, data['Close'], label='Actual Price', alpha=0.7)
                
                # Add prediction line if available
                if 'y_pred_test' in results and 'X_test' in results:
                    X_test = results['X_test']
                    y_pred = results['y_pred_test']
                    
                    # Create prediction series
                    pred_series = pd.Series(y_pred, index=X_test.index)
                    ax.plot(pred_series.index, pred_series.values, 
                           label='ML Predictions', alpha=0.8, linewidth=2)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title('ML Model Predictions')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                timeseries_path = output_dir / 'timeseries_predictions.png'
                plt.savefig(timeseries_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(timeseries_path)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
        
        return plots
    
    def run_single_backtest(self, data_file: Path, strategy_file: Path) -> List[Dict[str, Any]]:
        """Run single ML backtest"""
        try:
            # Load data
            data = self.load_data(data_file)
            if data is None or data.empty:
                return []
            
            # Load strategy
            strategy_cls = self.load_strategy(strategy_file)
            if strategy_cls is None:
                return []
            
            # Create features
            features = self.create_features(data)
            if features.empty:
                return []
            
            # Prepare target variable (future returns)
            target = features['Close'].pct_change().shift(-self.config.prediction_horizon).dropna()
            features = features.iloc[:-self.config.prediction_horizon]
            
            # Align features and target
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]
            
            if len(features) < 100:  # Need sufficient data
                return []
            
            # Select features
            if len(features.columns) > self.config.max_features:
                selector = SelectKBest(f_regression, k=self.config.max_features)
                features_selected = selector.fit_transform(features, target)
                feature_names = features.columns[selector.get_support()]
                features = pd.DataFrame(features_selected, index=features.index, columns=feature_names)
            
            # Train model
            model_results = self.train_model(features, target)
            
            if not model_results.get('success', False):
                return []
            
            # Create result
            result = {
                'data_file': data_file.name,
                'strategy_file': strategy_file.name,
                'engine': 'OptimizedMLEngine',
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config.model_type.value,
                'train_r2': model_results['train_r2'],
                'test_r2': model_results['test_r2'],
                'cv_mean': model_results['cv_mean'],
                'cv_std': model_results['cv_std'],
                'train_mse': model_results['train_mse'],
                'test_mse': model_results['test_mse'],
                'feature_count': len(features.columns),
                'sample_count': len(features)
            }
            
            # Create visualizations
            if self.config.generate_plots:
                output_dir = self.get_results_directory() / 'visualizations' / f"{data_file.stem}_{strategy_file.stem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                plots = self.create_visualizations(data, model_results, output_dir)
                result['visualizations'] = [str(p) for p in plots]
            
            return [result]
            
        except Exception as e:
            self.logger.error(f"Error in ML backtest: {e}")
            return []
    
    def run(self):
        """Main execution method"""
        if not ML_AVAILABLE:
            self.logger.error("ML libraries not available. Install scikit-learn.")
            return
        
        # Discover files
        data_files = self.discover_data_files()
        strategy_files = self.discover_strategy_files()
        
        if not data_files or not strategy_files:
            self.logger.error("No data files or strategy files found")
            return
        
        # Process combinations
        all_results = []
        total_combinations = len(data_files) * len(strategy_files)
        
        self.logger.info(f"Processing {total_combinations} ML backtest combinations...")
        
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
            self.logger.info(f"ML Backtest Summary:")
            self.logger.info(f"   • Total combinations: {len(all_results)}")
            self.logger.info(f"   • Strategies tested: {df['strategy_file'].nunique()}")
            self.logger.info(f"   • Data files tested: {df['data_file'].nunique()}")
            self.logger.info(f"   • Average test R²: {df['test_r2'].mean():.3f}")
            self.logger.info(f"   • Best test R²: {df['test_r2'].max():.3f}")
            self.logger.info(f"   • Average CV score: {df['cv_mean'].mean():.3f}")
    
    def save_engine_specific_formats(self, result: Dict[str, Any], data_dir: Path):
        """Save ML-specific formats"""
        try:
            # Save model performance metrics as CSV
            performance_data = {
                'Metric': ['Train R²', 'Test R²', 'CV Mean', 'CV Std', 'Train MSE', 'Test MSE', 
                          'Feature Count', 'Sample Count'],
                'Value': [
                    result.get('train_r2', 0),
                    result.get('test_r2', 0),
                    result.get('cv_mean', 0),
                    result.get('cv_std', 0),
                    result.get('train_mse', 0),
                    result.get('test_mse', 0),
                    result.get('feature_count', 0),
                    result.get('sample_count', 0)
                ]
            }
            
            performance_df = pd.DataFrame(performance_data)
            performance_path = data_dir / f"{data_dir.name}_model_performance.csv"
            performance_df.to_csv(performance_path, index=False)
            
            # Save model details
            model_details = {
                'model_type': result.get('model_type', 'unknown'),
                'feature_engineering': [fe.value for fe in self.config.feature_engineering],
                'test_size': self.config.test_size,
                'cv_folds': self.config.cross_validation_folds,
                'prediction_horizon': self.config.prediction_horizon,
                'max_features': self.config.max_features
            }
            
            model_path = data_dir / f"{data_dir.name}_model_details.json"
            with open(model_path, 'w') as f:
                json.dump(model_details, f, indent=2)
            
            # Save feature importance if available (would need to be passed from training)
            if 'feature_importance' in result and result['feature_importance']:
                importance_df = pd.DataFrame(list(result['feature_importance'].items()),
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False)
                importance_path = data_dir / f"{data_dir.name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving ML-specific formats: {e}")

def main():
    """Main entry point"""
    config = MLEngineConfig()
    engine = OptimizedMLEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
