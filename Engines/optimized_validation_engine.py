#!/usr/bin/env python3
"""
Optimized Validation Engine - Based on old_engines patterns
Incorporates robust error handling, parallel processing, and comprehensive validation
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from scipy import stats
import importlib.util
import inspect

# Suppress warnings
warnings.filterwarnings("ignore", message="A contingent SL/TP order would execute in the same bar*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backtesting\._stats")

@dataclass
class ValidationEngineConfig:
    """Configuration for validation engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Backtesting parameters
    initial_cash: float = 10000.0
    commission: float = 0.002
    backtest_timeout: int = 300
    
    # Validation parameters
    n_splits: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    n_bootstrap_samples: int = 100
    n_permutations: int = 50
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    # Walk-forward parameters
    walkforward_train_size: int = 700
    walkforward_test_size: int = 300
    walkforward_step_size: int = 300
    walkforward_min_total_size: int = 300
    
    # Performance optimization
    parallel_workers: int = 4
    skip_existing_results: bool = True
    
    # Results directory handling
    results_subdir_prefix: str = "validation_backtest"
    
    # Output options
    save_json: bool = True
    save_csv: bool = True
    log_level: str = "INFO"

class ValidationEngine:
    """Validation engine with comprehensive testing methods"""
    
    def __init__(self, config: ValidationEngineConfig = None):
        self.config = config or ValidationEngineConfig()
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
                logging.FileHandler(f'validation_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    
    def run_cross_validation(self, strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run cross-validation analysis"""
        try:
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            cv_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
                if self.interrupted:
                    break
                
                train_data = data.iloc[train_idx].copy()
                test_data = data.iloc[test_idx].copy()
                
                # Run backtest on training data
                train_result = self.run_single_backtest(strategy_cls(), train_data)
                if not train_result:
                    continue
                
                # Run backtest on test data
                test_result = self.run_single_backtest(strategy_cls(), test_data)
                if not test_result:
                    continue
                
                cv_results.append({
                    'fold': fold,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_return': train_result['total_return'],
                    'test_return': test_result['total_return'],
                    'train_sharpe': train_result['sharpe_ratio'],
                    'test_sharpe': test_result['sharpe_ratio'],
                    'train_drawdown': train_result['max_drawdown'],
                    'test_drawdown': test_result['max_drawdown']
                })
            
            if not cv_results:
                return {}
            
            df_cv = pd.DataFrame(cv_results)
            
            return {
                'cross_validation': {
                    'n_folds': len(cv_results),
                    'avg_train_return': df_cv['train_return'].mean(),
                    'avg_test_return': df_cv['test_return'].mean(),
                    'avg_train_sharpe': df_cv['train_sharpe'].mean(),
                    'avg_test_sharpe': df_cv['test_sharpe'].mean(),
                    'return_consistency': df_cv['test_return'].std(),
                    'sharpe_consistency': df_cv['test_sharpe'].std(),
                    'overfitting_ratio': df_cv['train_return'].mean() / df_cv['test_return'].mean() if df_cv['test_return'].mean() != 0 else 0,
                    'best_fold': df_cv.loc[df_cv['test_return'].idxmax(), 'fold'],
                    'worst_fold': df_cv.loc[df_cv['test_return'].idxmin(), 'fold']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def run_walkforward_analysis(self, strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        try:
            total_size = len(data)
            if total_size < self.config.walkforward_min_total_size:
                return {}
            
            window_size = self.config.walkforward_train_size + self.config.walkforward_test_size
            max_windows = min(
                (total_size - window_size) // self.config.walkforward_step_size + 1,
                10  # Limit to 10 windows for performance
            )
            
            if max_windows <= 0:
                return {}
            
            wf_results = []
            
            for i in range(max_windows):
                if self.interrupted:
                    break
                
                # Calculate window boundaries
                start_idx = i * self.config.walkforward_step_size
                train_end = start_idx + self.config.walkforward_train_size
                test_end = train_end + self.config.walkforward_test_size
                
                if test_end > total_size:
                    break
                
                # Split data
                train_data = data.iloc[start_idx:train_end].copy()
                test_data = data.iloc[train_end:test_end].copy()
                
                # Run backtest on training data
                train_result = self.run_single_backtest(strategy_cls(), train_data)
                if not train_result:
                    continue
                
                # Run backtest on test data
                test_result = self.run_single_backtest(strategy_cls(), test_data)
                if not test_result:
                    continue
                
                wf_results.append({
                    'window': i,
                    'train_start': start_idx,
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': test_end,
                    'train_return': train_result['total_return'],
                    'test_return': test_result['total_return'],
                    'train_sharpe': train_result['sharpe_ratio'],
                    'test_sharpe': test_result['sharpe_ratio'],
                    'train_drawdown': train_result['max_drawdown'],
                    'test_drawdown': test_result['max_drawdown']
                })
            
            if not wf_results:
                return {}
            
            df_wf = pd.DataFrame(wf_results)
            
            return {
                'walkforward_analysis': {
                    'n_windows': len(wf_results),
                    'avg_train_return': df_wf['train_return'].mean(),
                    'avg_test_return': df_wf['test_return'].mean(),
                    'avg_train_sharpe': df_wf['train_sharpe'].mean(),
                    'avg_test_sharpe': df_wf['test_sharpe'].mean(),
                    'return_consistency': df_wf['test_return'].std(),
                    'sharpe_consistency': df_wf['test_sharpe'].std(),
                    'performance_degradation': (df_wf['train_return'].mean() - df_wf['test_return'].mean()) / abs(df_wf['train_return'].mean()) if df_wf['train_return'].mean() != 0 else 0,
                    'best_window': df_wf.loc[df_wf['test_return'].idxmax(), 'window'],
                    'worst_window': df_wf.loc[df_wf['test_return'].idxmin(), 'window']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {}
    
    def run_bootstrap_validation(self, strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run bootstrap validation"""
        try:
            bootstrap_results = []
            
            for i in range(self.config.n_bootstrap_samples):
                if self.interrupted:
                    break
                
                if i % 20 == 0:
                    self.logger.info(f"Bootstrap sample {i}/{self.config.n_bootstrap_samples}")
                
                # Create bootstrap sample
                bootstrap_data = data.sample(n=len(data), replace=True).reset_index(drop=True)
                
                # Run backtest on bootstrap sample
                bootstrap_result = self.run_single_backtest(strategy_cls(), bootstrap_data)
                if bootstrap_result:
                    bootstrap_results.append(bootstrap_result['total_return'])
            
            if not bootstrap_results:
                return {}
            
            bootstrap_returns = np.array(bootstrap_results)
            
            # Calculate confidence intervals
            alpha = 1 - self.config.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_returns, lower_percentile)
            ci_upper = np.percentile(bootstrap_returns, upper_percentile)
            
            return {
                'bootstrap_validation': {
                    'bootstrap_mean_return': np.mean(bootstrap_returns),
                    'bootstrap_std_return': np.std(bootstrap_returns),
                    'confidence_interval_lower': ci_lower,
                    'confidence_interval_upper': ci_upper,
                    'confidence_level': self.config.confidence_level,
                    'n_bootstrap_samples': len(bootstrap_returns),
                    'bootstrap_skewness': stats.skew(bootstrap_returns),
                    'bootstrap_kurtosis': stats.kurtosis(bootstrap_returns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap validation: {e}")
            return {}
    
    def run_permutation_test(self, strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run permutation test for significance"""
        try:
            # Run backtest on original data
            original_result = self.run_single_backtest(strategy_cls(), data)
            if not original_result:
                return {}
            
            original_return = original_result['total_return']
            permutation_returns = []
            
            for i in range(self.config.n_permutations):
                if self.interrupted:
                    break
                
                if i % 10 == 0:
                    self.logger.info(f"Permutation {i}/{self.config.n_permutations}")
                
                # Create random data by shuffling returns
                random_data = data.copy()
                returns = data['Close'].pct_change().dropna()
                shuffled_returns = np.random.permutation(returns.values)
                
                # Reconstruct price series
                random_prices = [data['Close'].iloc[0]]
                for ret in shuffled_returns:
                    random_prices.append(random_prices[-1] * (1 + ret))
                
                random_data['Close'] = random_prices[:len(random_data)]
                
                # Run backtest on random data
                random_result = self.run_single_backtest(strategy_cls(), random_data)
                if random_result:
                    permutation_returns.append(random_result['total_return'])
            
            if not permutation_returns:
                return {}
            
            permutation_returns = np.array(permutation_returns)
            p_value = np.mean(permutation_returns >= original_return)
            
            return {
                'permutation_test': {
                    'p_value': p_value,
                    'is_significant': p_value < self.config.significance_level,
                    'original_return': original_return,
                    'permutation_mean_return': np.mean(permutation_returns),
                    'permutation_std_return': np.std(permutation_returns),
                    'n_permutations': len(permutation_returns),
                    'significance_level': self.config.significance_level
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in permutation test: {e}")
            return {}
    
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
            
            # Run validation tests
            validation_results = {}
            
            # Cross-validation
            cv_results = self.run_cross_validation(strategy_cls, data)
            validation_results.update(cv_results)
            
            # Walk-forward analysis
            wf_results = self.run_walkforward_analysis(strategy_cls, data)
            validation_results.update(wf_results)
            
            # Bootstrap validation
            bootstrap_results = self.run_bootstrap_validation(strategy_cls, data)
            validation_results.update(bootstrap_results)
            
            # Permutation test
            permutation_results = self.run_permutation_test(strategy_cls, data)
            validation_results.update(permutation_results)
            
            # Combine all results
            final_result = validation_results.copy()
            final_result.update({
                'strategy_name': strategy_name,
                'data_file': data_file_name,
                'engine_name': 'ValidationEngine',
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
        
        # Save validation summary
        self.save_validation_summary(results, results_dir)
    
    def save_validation_summary(self, results: List[Dict[str, Any]], results_dir: Path):
        """Save validation analysis summary"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Validation metrics summary
        validation_summary = {
            'total_combinations': len(results),
            'strategies_tested': df['strategy_name'].nunique(),
            'data_files_tested': df['data_file'].nunique(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cross-validation summary
        if 'cross_validation.avg_test_return' in df.columns:
            cv_data = df[df['cross_validation.avg_test_return'].notna()]
            if not cv_data.empty:
                validation_summary['cross_validation_analysis'] = {
                    'combinations_with_cv': len(cv_data),
                    'avg_test_return': cv_data['cross_validation.avg_test_return'].mean(),
                    'avg_overfitting_ratio': cv_data['cross_validation.overfitting_ratio'].mean(),
                    'avg_return_consistency': cv_data['cross_validation.return_consistency'].mean()
                }
        
        # Walk-forward analysis summary
        if 'walkforward_analysis.avg_test_return' in df.columns:
            wf_data = df[df['walkforward_analysis.avg_test_return'].notna()]
            if not wf_data.empty:
                validation_summary['walkforward_analysis'] = {
                    'combinations_with_wf': len(wf_data),
                    'avg_test_return': wf_data['walkforward_analysis.avg_test_return'].mean(),
                    'avg_performance_degradation': wf_data['walkforward_analysis.performance_degradation'].mean(),
                    'avg_return_consistency': wf_data['walkforward_analysis.return_consistency'].mean()
                }
        
        # Bootstrap analysis summary
        if 'bootstrap_validation.bootstrap_mean_return' in df.columns:
            bootstrap_data = df[df['bootstrap_validation.bootstrap_mean_return'].notna()]
            if not bootstrap_data.empty:
                validation_summary['bootstrap_analysis'] = {
                    'combinations_with_bootstrap': len(bootstrap_data),
                    'avg_bootstrap_mean': bootstrap_data['bootstrap_validation.bootstrap_mean_return'].mean(),
                    'avg_ci_width': (bootstrap_data['bootstrap_validation.confidence_interval_upper'] - 
                                   bootstrap_data['bootstrap_validation.confidence_interval_lower']).mean()
                }
        
        # Permutation test summary
        if 'permutation_test.p_value' in df.columns:
            significant_count = df[df['permutation_test.is_significant'] == True].shape[0]
            validation_summary['permutation_analysis'] = {
                'significant_strategies': significant_count,
                'significance_rate': significant_count / len(results),
                'avg_p_value': df['permutation_test.p_value'].mean()
            }
        
        summary_path = results_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        self.logger.info(f"📊 Validation Summary: {validation_summary['total_combinations']} combinations, "
                        f"significance rate: {validation_summary.get('permutation_analysis', {}).get('significance_rate', 0):.2%}")
    
    def run(self):
        """Main execution method"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Validation Engine...")
        
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
        self.logger.info(f"✅ Validation analysis complete! {len(all_results)} results in {execution_time}")

def main():
    """Main entry point"""
    config = ValidationEngineConfig()
    engine = ValidationEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
