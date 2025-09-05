#!/usr/bin/env python3
"""
Improved Statistical Engine - Based on old_engines statistical_engine patterns
Incorporates permutation tests, bootstrap analysis, Monte Carlo simulations, and market regime detection
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import signal
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from backtesting import Backtest
from sklearn.utils import resample
from sklearn.cluster import KMeans
import importlib.util
import inspect

# Suppress warnings
warnings.filterwarnings("ignore", message="A contingent SL/TP order would execute in the same bar*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backtesting\._stats")

@dataclass
class StatisticalEngineConfig:
    """Configuration for statistical engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Backtesting parameters
    initial_cash: float = 10000.0
    commission: float = 0.002
    backtest_timeout: int = 300
    
    # Statistical testing parameters
    n_permutations: int = 50
    n_bootstrap_samples: int = 30
    n_monte_carlo_simulations: int = 25
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    # Test types
    run_permutation_tests: bool = True
    run_bootstrap_tests: bool = True
    run_monte_carlo_tests: bool = True
    run_gbm_mc: bool = True
    gbm_simulations: int = 50
    gbm_annual_drift: Optional[float] = None
    gbm_annual_vol: Optional[float] = None
    
    # Market regime analysis
    enable_regime_analysis: bool = True
    regime_window_size: int = 50
    n_regimes: int = 3
    
    # Performance optimization
    parallel_workers: int = 4
    skip_existing_results: bool = True
    
    # Results directory handling
    results_subdir_prefix: str = "improved_statistical_backtest"
    
    # Output options
    save_json: bool = True
    save_csv: bool = True
    log_level: str = "INFO"

class StatisticalEngine:
    """Statistical engine with advanced analysis"""
    
    def __init__(self, config: StatisticalEngineConfig = None):
        self.config = config or StatisticalEngineConfig()
        self.setup_logging()
        self.setup_signal_handlers()
        self.results = []
        self.interrupted = False
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'improved_statistical_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info("🛑 Interrupt signal received. Gracefully shutting down...")
            self.interrupted = True
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def create_random_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create random data that preserves some market properties"""
        random_data = original_data.copy()
        
        # Preserve price levels but randomize returns
        returns = original_data['Close'].pct_change().dropna()
        
        # Bootstrap returns to preserve some market properties
        random_returns = resample(returns, n_samples=len(returns), 
                                random_state=np.random.randint(1000))
        
        # Reconstruct price series
        reconstructed_close = original_data['Close'].iloc[0] * (1 + np.array(random_returns)).cumprod()
        
        # Ensure the reconstructed series has the same length as original
        if len(reconstructed_close) < len(original_data):
            padding = [reconstructed_close[-1]] * (len(original_data) - len(reconstructed_close))
            reconstructed_close = np.concatenate([reconstructed_close, padding])
        
        random_data['Close'] = reconstructed_close[:len(original_data)]
        
        # Recalculate other columns based on new close prices
        if 'High' in random_data.columns:
            random_data['High'] = random_data['Close'] * (1 + np.random.uniform(0, 0.02, len(random_data)))
        if 'Low' in random_data.columns:
            random_data['Low'] = random_data['Close'] * (1 - np.random.uniform(0, 0.02, len(random_data)))
        if 'Open' in random_data.columns:
            random_data['Open'] = random_data['Close'] * (1 + np.random.uniform(-0.01, 0.01, len(random_data)))
        
        return random_data
    
    def run_permutation_test(self, original_result: Dict[str, Any], 
                           strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run permutation test for significance"""
        if not self.config.run_permutation_tests:
            return {}
        
        try:
            original_return = original_result['total_return']
            random_returns = []
            
            self.logger.info(f"Running {self.config.n_permutations} permutations...")
            
            for i in range(self.config.n_permutations):
                if self.interrupted:
                    break
                
                if i % 10 == 0:
                    self.logger.info(f"Permutation {i}/{self.config.n_permutations}")
                
                # Create random data
                random_data = self.create_random_data(data)
                
                # Run backtest on random data
                random_result = self.run_single_backtest(strategy_cls(), random_data)
                if random_result:
                    random_returns.append(random_result['total_return'])
            
            if random_returns:
                random_returns = np.array(random_returns)
                p_value = np.mean(random_returns >= original_return)
                
                return {
                    'permutation_test': {
                        'p_value': p_value,
                        'is_significant': p_value < self.config.significance_level,
                        'random_mean_return': np.mean(random_returns),
                        'random_std_return': np.std(random_returns),
                        'n_permutations': len(random_returns),
                        'original_return': original_return
                    }
                }
            
        except Exception as e:
            self.logger.error(f"Error in permutation test: {e}")
        
        return {}
    
    def run_bootstrap_test(self, original_result: Dict[str, Any], 
                         strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run bootstrap test for confidence intervals"""
        if not self.config.run_bootstrap_tests:
            return {}
        
        try:
            bootstrap_returns = []
            
            self.logger.info(f"Running {self.config.n_bootstrap_samples} bootstrap samples...")
            
            for i in range(self.config.n_bootstrap_samples):
                if self.interrupted:
                    break
                
                if i % 10 == 0:
                    self.logger.info(f"Bootstrap {i}/{self.config.n_bootstrap_samples}")
                
                # Create bootstrap sample
                bootstrap_data = data.sample(n=len(data), replace=True).reset_index(drop=True)
                
                # Run backtest on bootstrap sample
                bootstrap_result = self.run_single_backtest(strategy_cls(), bootstrap_data)
                if bootstrap_result:
                    bootstrap_returns.append(bootstrap_result['total_return'])
            
            if bootstrap_returns:
                bootstrap_returns = np.array(bootstrap_returns)
                
                # Calculate confidence intervals
                alpha = 1 - self.config.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                ci_lower = np.percentile(bootstrap_returns, lower_percentile)
                ci_upper = np.percentile(bootstrap_returns, upper_percentile)
                
                return {
                    'bootstrap_test': {
                        'bootstrap_mean_return': np.mean(bootstrap_returns),
                        'bootstrap_std_return': np.std(bootstrap_returns),
                        'confidence_interval_lower': ci_lower,
                        'confidence_interval_upper': ci_upper,
                        'confidence_level': self.config.confidence_level,
                        'n_bootstrap_samples': len(bootstrap_returns),
                        'original_return': original_result['total_return'],
                        'is_within_ci': ci_lower <= original_result['total_return'] <= ci_upper
                    }
                }
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap test: {e}")
        
        return {}
    
    def run_monte_carlo_test(self, original_result: Dict[str, Any], 
                           strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run Monte Carlo simulation test"""
        if not self.config.run_monte_carlo_tests:
            return {}
        
        try:
            monte_carlo_returns = []
            
            self.logger.info(f"Running {self.config.n_monte_carlo_simulations} Monte Carlo simulations...")
            
            for i in range(self.config.n_monte_carlo_simulations):
                if self.interrupted:
                    break
                
                if i % 10 == 0:
                    self.logger.info(f"Monte Carlo {i}/{self.config.n_monte_carlo_simulations}")
                
                # Create Monte Carlo data
                mc_data = self.create_monte_carlo_data(data)
                
                # Run backtest on Monte Carlo data
                mc_result = self.run_single_backtest(strategy_cls(), mc_data)
                if mc_result:
                    monte_carlo_returns.append(mc_result['total_return'])
            
            if monte_carlo_returns:
                monte_carlo_returns = np.array(monte_carlo_returns)
                
                return {
                    'monte_carlo_test': {
                        'mc_mean_return': np.mean(monte_carlo_returns),
                        'mc_std_return': np.std(monte_carlo_returns),
                        'mc_percentile_rank': np.percentile(monte_carlo_returns, 
                                                           original_result['total_return'] * 100),
                        'n_simulations': len(monte_carlo_returns),
                        'original_return': original_result['total_return']
                    }
                }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo test: {e}")
        
        return {}
    
    def create_monte_carlo_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create Monte Carlo simulated data"""
        mc_data = original_data.copy()
        
        # Calculate returns
        returns = original_data['Close'].pct_change().dropna()
        
        # Fit normal distribution to returns
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random returns from normal distribution
        random_returns = np.random.normal(mu, sigma, len(returns))
        
        # Reconstruct price series
        reconstructed_close = original_data['Close'].iloc[0] * (1 + random_returns).cumprod()
        
        # Ensure the reconstructed series has the same length as original
        if len(reconstructed_close) < len(original_data):
            padding = [reconstructed_close[-1]] * (len(original_data) - len(reconstructed_close))
            reconstructed_close = np.concatenate([reconstructed_close, padding])
        
        mc_data['Close'] = reconstructed_close[:len(original_data)]
        
        # Recalculate other columns
        if 'High' in mc_data.columns:
            mc_data['High'] = mc_data['Close'] * (1 + np.random.uniform(0, 0.02, len(mc_data)))
        if 'Low' in mc_data.columns:
            mc_data['Low'] = mc_data['Close'] * (1 - np.random.uniform(0, 0.02, len(mc_data)))
        if 'Open' in mc_data.columns:
            mc_data['Open'] = mc_data['Close'] * (1 + np.random.uniform(-0.01, 0.01, len(mc_data)))
        
        return mc_data
    
    def run_gbm_monte_carlo(self, original_result: Dict[str, Any], 
                          data: pd.DataFrame) -> Dict[str, Any]:
        """Run Geometric Brownian Motion Monte Carlo simulation"""
        if not self.config.run_gbm_mc:
            return {}
        
        try:
            # Calculate annual drift and volatility from data
            returns = data['Close'].pct_change().dropna()
            annual_drift = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            
            # Use provided values if available
            if self.config.gbm_annual_drift is not None:
                annual_drift = self.config.gbm_annual_drift
            if self.config.gbm_annual_vol is not None:
                annual_vol = self.config.gbm_annual_vol
            
            gbm_returns = []
            
            self.logger.info(f"Running {self.config.gbm_simulations} GBM simulations...")
            
            for i in range(self.config.gbm_simulations):
                if self.interrupted:
                    break
                
                if i % 10 == 0:
                    self.logger.info(f"GBM {i}/{self.config.gbm_simulations}")
                
                # Generate GBM path
                dt = 1/252  # Daily time step
                n_steps = len(data)
                
                # Random walk
                random_shocks = np.random.normal(0, 1, n_steps)
                
                # GBM formula: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
                log_returns = (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * random_shocks
                
                # Reconstruct price series
                gbm_prices = data['Close'].iloc[0] * np.exp(np.cumsum(log_returns))
                
                # Calculate total return
                gbm_return = (gbm_prices[-1] / gbm_prices[0]) - 1
                gbm_returns.append(gbm_return)
            
            if gbm_returns:
                gbm_returns = np.array(gbm_returns)
                
                return {
                    'gbm_monte_carlo': {
                        'gbm_mean_return': np.mean(gbm_returns),
                        'gbm_std_return': np.std(gbm_returns),
                        'gbm_percentile_rank': np.percentile(gbm_returns, 
                                                            original_result['total_return'] * 100),
                        'annual_drift': annual_drift,
                        'annual_vol': annual_vol,
                        'n_simulations': len(gbm_returns),
                        'original_return': original_result['total_return']
                    }
                }
            
        except Exception as e:
            self.logger.error(f"Error in GBM Monte Carlo: {e}")
        
        return {}
    
    def run_regime_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run market regime analysis"""
        if not self.config.enable_regime_analysis:
            return {}
        
        try:
            # Calculate market features
            features = self.calculate_market_features(data)
            
            if len(features) < self.config.regime_window_size:
                return {}
            
            # Prepare features for clustering
            feature_cols = ['volatility', 'trend_strength', 'volume_profile', 'momentum', 'mean_reversion']
            available_features = [col for col in feature_cols if col in features.columns]
            
            if not available_features:
                return {}
            
            X = features[available_features].dropna()
            
            if len(X) < self.config.n_regimes:
                return {}
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(X)
            
            # Analyze regime characteristics
            regime_analysis = {}
            for i in range(self.config.n_regimes):
                regime_mask = regime_labels == i
                regime_data = X[regime_mask]
                
                regime_analysis[f'regime_{i}'] = {
                    'count': np.sum(regime_mask),
                    'percentage': np.sum(regime_mask) / len(regime_labels) * 100,
                    'avg_volatility': regime_data['volatility'].mean() if 'volatility' in regime_data.columns else 0,
                    'avg_trend_strength': regime_data['trend_strength'].mean() if 'trend_strength' in regime_data.columns else 0,
                    'avg_volume_profile': regime_data['volume_profile'].mean() if 'volume_profile' in regime_data.columns else 0
                }
            
            return {
                'regime_analysis': {
                    'n_regimes': self.config.n_regimes,
                    'regime_characteristics': regime_analysis,
                    'regime_labels': regime_labels.tolist(),
                    'feature_importance': dict(zip(available_features, kmeans.feature_importances_)) if hasattr(kmeans, 'feature_importances_') else {}
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in regime analysis: {e}")
        
        return {}
    
    def calculate_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market features for regime analysis"""
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
        
        # Clean features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def run_single_backtest(self, strategy_instance, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run a single backtest"""
        try:
            # Prepare data
            data_for_backtest = data.copy()
            data_for_backtest.columns = [col.lower() for col in data_for_backtest.columns]
            
            # Run backtest
            bt = Backtest(data_for_backtest, strategy_instance,
                         cash=self.config.initial_cash,
                         commission=self.config.commission)
            
            stats = bt.run()
            
            # Extract metrics
            result = {
                'total_return': stats['Return [%]'] / 100,
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': abs(stats['Max. Drawdown [%]']) / 100,
                'win_rate': stats['Win Rate [%]'] / 100,
                'profit_factor': stats['Profit Factor'],
                'num_trades': stats['# Trades'],
                'volatility': stats['Volatility (Ann.) [%]'] / 100,
                'calmar_ratio': stats['Calmar Ratio'],
                'sortino_ratio': stats['Sortino Ratio']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single backtest: {e}")
            return None
    
    def discover_data_files(self) -> List[Path]:
        """Discover all CSV data files"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            self.logger.error(f"❌ Data path does not exist: {data_path}")
            return []
        
        data_files = list(data_path.glob("*.csv"))
        self.logger.info(f"📊 Found {len(data_files)} data files")
        return data_files
    
    def discover_strategy_files(self) -> List[Path]:
        """Discover all strategy files"""
        strategies_path = Path(self.config.strategies_path)
        if not strategies_path.exists():
            self.logger.error(f"❌ Strategies path does not exist: {strategies_path}")
            return []
        
        strategy_files = [f for f in strategies_path.glob("*.py") 
                         if f.name not in ["__init__.py", "strategy_factory.py", "base_strategy.py"]]
        self.logger.info(f"🎯 Found {len(strategy_files)} strategy files")
        return strategy_files
    
    def load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and validate data file"""
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns in {file_path.name}: {missing_columns}")
                return None
            
            # Clean data
            df = df.dropna()
            df = df[df['Volume'] > 0]
            
            if len(df) < 50:
                self.logger.warning(f"Insufficient data in {file_path.name}: {len(df)} rows")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path.name}: {e}")
            return None
    
    def load_strategy(self, strategy_file: Path):
        """Load strategy class from file"""
        try:
            spec = importlib.util.spec_from_file_location(strategy_file.stem, strategy_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[strategy_file.stem] = module
            spec.loader.exec_module(module)
            
            # Find strategy class
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'next') and 
                    obj.__module__ == module.__name__):
                    return obj
            
            self.logger.warning(f"No strategy class found in {strategy_file.name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading strategy from {strategy_file.name}: {e}")
            return None
    
    def process_file_strategy_combination(self, data_file: Path, strategy_file: Path) -> List[Dict[str, Any]]:
        """Process a single data file with a single strategy"""
        results = []
        
        try:
            # Load data
            data = self.load_data(data_file)
            if data is None:
                return results
            
            # Load strategy
            strategy_cls = self.load_strategy(strategy_file)
            if strategy_cls is None:
                return results
            
            strategy_name = strategy_file.stem
            data_file_name = data_file.stem
            
            # Run base backtest
            base_result = self.run_single_backtest(strategy_cls(), data)
            if base_result is None:
                return results
            
            # Run statistical tests
            statistical_results = {}
            
            # Permutation test
            permutation_results = self.run_permutation_test(base_result, strategy_cls, data)
            statistical_results.update(permutation_results)
            
            # Bootstrap test
            bootstrap_results = self.run_bootstrap_test(base_result, strategy_cls, data)
            statistical_results.update(bootstrap_results)
            
            # Monte Carlo test
            monte_carlo_results = self.run_monte_carlo_test(base_result, strategy_cls, data)
            statistical_results.update(monte_carlo_results)
            
            # GBM Monte Carlo
            gbm_results = self.run_gbm_monte_carlo(base_result, data)
            statistical_results.update(gbm_results)
            
            # Regime analysis
            regime_results = self.run_regime_analysis(data)
            statistical_results.update(regime_results)
            
            # Combine all results
            final_result = base_result.copy()
            final_result.update(statistical_results)
            final_result.update({
                'strategy_name': strategy_name,
                'data_file': data_file_name,
                'engine_name': 'ImprovedStatisticalEngine',
                'timestamp': datetime.now().isoformat()
            })
            
            results.append(final_result)
            
        except Exception as e:
            self.logger.error(f"Error processing {strategy_file.name} on {data_file.name}: {e}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to files"""
        if not results:
            self.logger.warning("No results to save")
            return
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_path) / f"{self.config.results_subdir_prefix}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        if self.config.save_json:
            json_path = results_dir / "all_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"✅ Results saved to: {json_path}")
        
        # Save CSV
        if self.config.save_csv:
            df_results = pd.DataFrame(results)
            csv_path = results_dir / "all_results.csv"
            df_results.to_csv(csv_path, index=False)
            self.logger.info(f"✅ Results saved to: {csv_path}")
        
        # Save statistical analysis summary
        self.save_statistical_summary(results, results_dir)
    
    def save_statistical_summary(self, results: List[Dict[str, Any]], results_dir: Path):
        """Save statistical analysis summary"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Statistical metrics summary
        statistical_summary = {
            'total_combinations': len(results),
            'strategies_tested': df['strategy_name'].nunique(),
            'data_files_tested': df['data_file'].nunique(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Significance test summary
        if 'permutation_test.p_value' in df.columns:
            significant_count = df[df['permutation_test.p_value'] < self.config.significance_level].shape[0]
            statistical_summary['permutation_analysis'] = {
                'significant_strategies': significant_count,
                'significance_rate': significant_count / len(results),
                'avg_p_value': df['permutation_test.p_value'].mean()
            }
        
        # Bootstrap analysis summary
        if 'bootstrap_test.confidence_interval_lower' in df.columns:
            within_ci_count = df[df['bootstrap_test.is_within_ci'] == True].shape[0]
            statistical_summary['bootstrap_analysis'] = {
                'within_confidence_interval': within_ci_count,
                'ci_success_rate': within_ci_count / len(results),
                'avg_ci_width': (df['bootstrap_test.confidence_interval_upper'] - 
                               df['bootstrap_test.confidence_interval_lower']).mean()
            }
        
        # Regime analysis summary
        if 'regime_analysis.n_regimes' in df.columns:
            regime_data = df[df['regime_analysis.n_regimes'].notna()]
            if not regime_data.empty:
                statistical_summary['regime_analysis'] = {
                    'combinations_with_regime_analysis': len(regime_data),
                    'avg_regimes_detected': regime_data['regime_analysis.n_regimes'].mean()
                }
        
        summary_path = results_dir / "statistical_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(statistical_summary, f, indent=2)
        
        self.logger.info(f"📊 Statistical Summary: {statistical_summary['total_combinations']} combinations, "
                        f"significance rate: {statistical_summary.get('permutation_analysis', {}).get('significance_rate', 0):.2%}")
    
    def run(self):
        """Main execution method"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Improved Statistical Engine...")
        
        # Discover files
        data_files = self.discover_data_files()
        strategy_files = self.discover_strategy_files()
        
        if not data_files or not strategy_files:
            self.logger.error("❌ No data files or strategy files found")
            return
        
        total_combinations = len(data_files) * len(strategy_files)
        self.logger.info(f"📊 Total combinations to process: {total_combinations}")
        
        # Process combinations
        all_results = []
        completed = 0
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            future_to_combo = {}
            for data_file in data_files:
                for strategy_file in strategy_files:
                    if self.interrupted:
                        break
                    future = executor.submit(self.process_file_strategy_combination, data_file, strategy_file)
                    future_to_combo[future] = (data_file.name, strategy_file.name)
            
            # Process completed tasks
            for future in as_completed(future_to_combo):
                if self.interrupted:
                    break
                
                data_file_name, strategy_file_name = future_to_combo[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    completed += 1
                    
                    if completed % 5 == 0:
                        progress = (completed / total_combinations) * 100
                        self.logger.info(f"📈 Progress: {progress:.1f}% ({completed}/{total_combinations})")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {strategy_file_name} on {data_file_name}: {e}")
        
        # Save results
        self.save_results(all_results)
        
        execution_time = datetime.now() - start_time
        self.logger.info(f"✅ Statistical analysis complete! {len(all_results)} results in {execution_time}")

def main():
    """Main entry point"""
    config = StatisticalEngineConfig()
    engine = StatisticalEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
