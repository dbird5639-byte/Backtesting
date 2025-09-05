#!/usr/bin/env python3
"""
Base Optimized Engine - Common functionality for all engines
Provides performance, reliability, and result management features
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
import importlib.util
import inspect
import threading
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="A contingent SL/TP order would execute in the same bar*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backtesting\._stats")
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide", category=RuntimeWarning)

@dataclass
class BaseEngineConfig:
    """Base configuration for all engines"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Backtesting parameters
    initial_cash: float = 10000.0
    commission: float = 0.002
    backtest_timeout: int = 300  # 5 minutes
    
    # Data handling
    min_data_points: int = 50
    max_data_points: int = 5000
    truncate_excess_data: bool = True
    truncate_side: str = "tail"  # "tail" or "head"
    
    # Performance optimization
    parallel_workers: int = 4
    skip_existing_results: bool = True
    batch_size: int = 10  # Process results in batches
    
    # Results management
    save_individual_results: bool = True  # Save per strategy per data file
    save_combined_results: bool = True    # Save combined results
    save_summary: bool = True            # Save summary statistics
    
    # Terminal output
    verbose: bool = True
    progress_interval: int = 5  # Show progress every N completions
    show_performance_stats: bool = True
    
    # Output options
    save_json: bool = True
    save_csv: bool = True
    log_level: str = "INFO"

class BaseOptimizedEngine:
    """Base class for all optimized engines with common functionality"""
    
    def __init__(self, config: BaseEngineConfig = None, engine_name: str = "BaseEngine"):
        self.config = config or BaseEngineConfig()
        self.engine_name = engine_name
        self.setup_logging()
        self.setup_signal_handlers()
        self.results = []
        self.interrupted = False
        self.start_time = None
        self.processed_count = 0
        self.total_combinations = 0
        self.performance_stats = {
            'total_time': 0,
            'avg_time_per_combination': 0,
            'combinations_per_second': 0,
            'memory_usage': 0
        }
        self._lock = threading.Lock()
    
    def setup_logging(self):
        """Setup logging configuration with performance tracking"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # File handler
        file_handler = logging.FileHandler(
            f'{self.engine_name.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(self.engine_name)
        self.logger.setLevel(log_level)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info("🛑 Interrupt signal received. Gracefully shutting down...")
            self.interrupted = True
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def print_progress(self, completed: int, total: int, additional_info: str = ""):
        """Print progress with performance statistics"""
        if not self.config.verbose:
            return
        
        progress_pct = (completed / total) * 100
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Calculate performance stats
        if elapsed_time > 0:
            combinations_per_second = completed / elapsed_time
            eta_seconds = (total - completed) / combinations_per_second if combinations_per_second > 0 else 0
            eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"
        else:
            combinations_per_second = 0
            eta_str = "calculating..."
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * completed // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Print progress
        print(f"\r🚀 {self.engine_name} | {bar} | {progress_pct:.1f}% | "
              f"{completed}/{total} | {combinations_per_second:.1f}/s | ETA: {eta_str} | {additional_info}", 
              end="", flush=True)
        
        if completed == total:
            print()  # New line when complete
    
    def discover_data_files(self) -> List[Path]:
        """Discover all CSV data files with performance optimization"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            self.logger.error(f"❌ Data path does not exist: {data_path}")
            return []
        
        data_files = list(data_path.glob("*.csv"))
        self.logger.info(f"Found {len(data_files)} data files")
        return data_files
    
    def discover_strategy_files(self) -> List[Path]:
        """Discover all strategy files with performance optimization"""
        strategies_path = Path(self.config.strategies_path)
        if not strategies_path.exists():
            self.logger.error(f"❌ Strategies path does not exist: {strategies_path}")
            return []
        
        strategy_files = [f for f in strategies_path.glob("*.py") 
                         if f.name not in ["__init__.py", "strategy_factory.py", "base_strategy.py"]]
        self.logger.info(f"Found {len(strategy_files)} strategy files")
        return strategy_files
    
    def load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and validate data file with performance optimization"""
        try:
            # Use pandas read_csv with optimized parameters
            df = pd.read_csv(file_path, low_memory=False)
            
            # Check required columns (case-insensitive)
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_columns_lower = [col.lower() for col in df.columns]
            missing_columns = [col for col in required_columns if col.lower() not in df_columns_lower]
            if missing_columns:
                self.logger.warning(f"Missing columns in {file_path.name}: {missing_columns}")
                self.logger.warning(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Clean data efficiently
            df = df.dropna()
            df = df[df['Volume'] > 0]
            
            # Set datetime index for backtesting
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Check data length
            if len(df) < self.config.min_data_points:
                self.logger.warning(f"Insufficient data in {file_path.name}: {len(df)} rows")
                return None
            
            # Truncate if too large
            if len(df) > self.config.max_data_points:
                if self.config.truncate_excess_data:
                    if self.config.truncate_side == "tail":
                        df = df.tail(self.config.max_data_points)
                    else:
                        df = df.head(self.config.max_data_points)
                    self.logger.debug(f"Truncated {file_path.name} to {len(df)} rows")
                else:
                    self.logger.warning(f"Data too large in {file_path.name}: {len(df)} rows")
                    return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path.name}: {e}")
            return None
    
    def load_strategy(self, strategy_file: Path):
        """Load strategy class from file with caching"""
        try:
            # Check if already loaded
            if hasattr(self, '_strategy_cache'):
                if strategy_file in self._strategy_cache:
                    return self._strategy_cache[strategy_file]
            else:
                self._strategy_cache = {}
            
            spec = importlib.util.spec_from_file_location(strategy_file.stem, strategy_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[strategy_file.stem] = module
            spec.loader.exec_module(module)
            
            # Find strategy class
            strategy_cls = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'next') and 
                    obj.__module__ == module.__name__):
                    strategy_cls = obj
                    break
            
            if strategy_cls is None:
                self.logger.warning(f"No strategy class found in {strategy_file.name}")
                return None
            
            # Cache the strategy class
            self._strategy_cache[strategy_file] = strategy_cls
            return strategy_cls
            
        except Exception as e:
            self.logger.error(f"Error loading strategy from {strategy_file.name}: {e}")
            return None
    
    def run_single_backtest(self, strategy_cls, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run a single backtest with timeout and error handling"""
        try:
            # Prepare data efficiently
            data_for_backtest = data.copy()
            data_for_backtest.columns = [col.lower() for col in data_for_backtest.columns]
            
            # Run backtest with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_backtest_core, strategy_cls, data_for_backtest)
                try:
                    result = future.result(timeout=self.config.backtest_timeout)
                    return result
                except FuturesTimeoutError:
                    self.logger.warning(f"Backtest timeout for {strategy_cls.__name__}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in single backtest: {e}")
            return None
    
    def _run_backtest_core(self, strategy_cls, data_for_backtest: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Core backtest execution"""
        try:
            bt = Backtest(data_for_backtest, strategy_cls,
                         cash=self.config.initial_cash,
                         commission=self.config.commission)
            
            stats = bt.run()
            
            # Extract key metrics efficiently
            result = {
                'total_return': stats['Return [%]'] / 100,
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': abs(stats['Max. Drawdown [%]']) / 100,
                'win_rate': stats['Win Rate [%]'] / 100,
                'profit_factor': stats['Profit Factor'],
                'num_trades': stats['# Trades'],
                'volatility': stats['Volatility (Ann.) [%]'] / 100,
                'calmar_ratio': stats['Calmar Ratio'],
                'sortino_ratio': stats['Sortino Ratio'],
                'execution_time': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in backtest core: {e}")
            return None
    
    def save_individual_results(self, results: List[Dict[str, Any]], results_dir: Path):
        """Save individual results per strategy per data file"""
        if not self.config.save_individual_results or not results:
            return
        
        # Group results by strategy and data file
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            strategy_name = result.get('strategy_name', 'unknown')
            data_file = result.get('data_file', 'unknown')
            grouped_results[strategy_name][data_file].append(result)
        
        # Save individual files
        for strategy_name, data_files in grouped_results.items():
            strategy_dir = results_dir / strategy_name
            strategy_dir.mkdir(exist_ok=True)
            
            for data_file, file_results in data_files.items():
                # Save JSON
                if self.config.save_json:
                    json_path = strategy_dir / f"{data_file}.json"
                    with open(json_path, 'w') as f:
                        json.dump(file_results, f, indent=2)
                
                # Save CSV
                if self.config.save_csv:
                    csv_path = strategy_dir / f"{data_file}.csv"
                    df = pd.DataFrame(file_results)
                    df.to_csv(csv_path, index=False)
        
        self.logger.info(f"✅ Individual results saved to: {results_dir}")
    
    def save_combined_results(self, results: List[Dict[str, Any]], results_dir: Path):
        """Save combined results"""
        if not self.config.save_combined_results or not results:
            return
        
        # Save JSON
        if self.config.save_json:
            json_path = results_dir / "all_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"✅ Combined results saved to: {json_path}")
        
        # Save CSV
        if self.config.save_csv:
            df_results = pd.DataFrame(results)
            csv_path = results_dir / "all_results.csv"
            df_results.to_csv(csv_path, index=False)
            self.logger.info(f"✅ Combined results saved to: {csv_path}")
    
    def save_summary(self, results: List[Dict[str, Any]], results_dir: Path):
        """Save summary statistics"""
        if not self.config.save_summary or not results:
            return
        
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = {
            'engine_name': self.engine_name,
            'total_combinations': len(results),
            'strategies_tested': df['strategy_name'].nunique() if 'strategy_name' in df.columns else 0,
            'data_files_tested': df['data_file'].nunique() if 'data_file' in df.columns else 0,
            'execution_time': time.time() - self.start_time if self.start_time else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add performance metrics if available
        if 'total_return' in df.columns:
            summary['performance_metrics'] = {
                'avg_return': df['total_return'].mean(),
                'best_return': df['total_return'].max(),
                'worst_return': df['total_return'].min(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'best_sharpe': df['sharpe_ratio'].max(),
                'avg_drawdown': df['max_drawdown'].mean(),
                'worst_drawdown': df['max_drawdown'].max(),
                'avg_win_rate': df['win_rate'].mean(),
                'avg_trades': df['num_trades'].mean()
            }
        
        # Add performance stats
        summary['performance_stats'] = self.performance_stats
        
        # Save summary
        summary_path = results_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Summary saved to: {summary_path}")
    
    def update_performance_stats(self, completed: int, total: int):
        """Update performance statistics"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            self.performance_stats.update({
                'total_time': elapsed_time,
                'avg_time_per_combination': elapsed_time / completed if completed > 0 else 0,
                'combinations_per_second': completed / elapsed_time if elapsed_time > 0 else 0
            })
    
    def run(self):
        """Main execution method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement the run method")
    
    def process_combinations(self, data_files: List[Path], strategy_files: List[Path]):
        """Process all combinations with performance optimization"""
        self.start_time = time.time()
        self.total_combinations = len(data_files) * len(strategy_files)
        
        self.logger.info(f"🚀 Starting {self.engine_name}...")
        self.logger.info(f"📊 Processing {self.total_combinations} combinations")
        self.logger.info(f"⚡ Using {self.config.parallel_workers} parallel workers")
        
        all_results = []
        completed = 0
        
        # Use parallel processing with progress tracking
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            future_to_combo = {}
            for data_file in data_files:
                for strategy_file in strategy_files:
                    if self.interrupted:
                        break
                    future = executor.submit(
                        self.process_file_strategy_combination, 
                        data_file, 
                        strategy_file
                    )
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
                    
                    # Update performance stats
                    self.update_performance_stats(completed, self.total_combinations)
                    
                    # Show progress
                    if completed % self.config.progress_interval == 0 or completed == self.total_combinations:
                        self.print_progress(completed, self.total_combinations, 
                                          f"Results: {len(all_results)}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {strategy_file_name} on {data_file_name}: {e}")
        
        return all_results
    
    def process_file_strategy_combination(self, data_file: Path, strategy_file: Path) -> List[Dict[str, Any]]:
        """Process a single data file with a single strategy - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_file_strategy_combination")
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save all results with comprehensive organization"""
        if not results:
            self.logger.warning("No results to save")
            return
        
        # Create results directory structure: Results/Engine(1,2,3,etc)/Strategy(1,2,3,etc)/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = Path(self.config.results_path)
        
        # Save individual results (per strategy per data file)
        self.save_individual_results_organized(results, base_results_dir, timestamp)
        
        # Save combined results
        self.save_combined_results(results, base_results_dir / f"{self.engine_name.lower()}_{timestamp}")
        
        execution_time = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"✅ {self.engine_name} complete! {len(results)} results in {execution_time:.2f}s")
        
        if self.config.show_performance_stats:
            self.logger.info(f"📈 Performance: {self.performance_stats['combinations_per_second']:.1f} combinations/second")
    
    def save_individual_results_organized(self, results: List[Dict[str, Any]], base_dir: Path, timestamp: str):
        """Save individual results in organized structure: Results/Engine(1,2,3,etc)/Strategy(1,2,3,etc)/"""
        try:
            # Group results by strategy
            strategy_groups = {}
            for result in results:
                strategy_name = result.get('strategy_file', 'unknown_strategy').replace('.py', '')
                if strategy_name not in strategy_groups:
                    strategy_groups[strategy_name] = []
                strategy_groups[strategy_name].append(result)
            
            # Create engine directory
            engine_dir = base_dir / f"Engine_{self.engine_name}_{timestamp}"
            engine_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each strategy's results
            for strategy_name, strategy_results in strategy_groups.items():
                # Create strategy directory
                strategy_dir = engine_dir / f"Strategy_{strategy_name}"
                strategy_dir.mkdir(parents=True, exist_ok=True)
                
                # Save each data file result
                for result in strategy_results:
                    data_file_name = result.get('data_file', 'unknown_data').replace('.csv', '')
                    data_dir = strategy_dir / f"Data_{data_file_name}"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save in multiple formats
                    self.save_result_formats(result, data_dir)
                
                # Save strategy summary
                self.save_strategy_summary(strategy_results, strategy_dir)
            
            self.logger.info(f"📁 Individual results saved to: {engine_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving organized results: {e}")
    
    def save_result_formats(self, result: Dict[str, Any], data_dir: Path):
        """Save result in multiple formats (.csv, .json, .png, etc.)"""
        try:
            # Save as JSON
            json_path = data_dir / f"{data_dir.name}_result.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Save as CSV (flattened)
            csv_path = data_dir / f"{data_dir.name}_metrics.csv"
            self.save_result_as_csv(result, csv_path)
            
            # Save visualizations if they exist
            if 'visualizations' in result and result['visualizations']:
                viz_dir = data_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                for viz_path in result['visualizations']:
                    if Path(viz_path).exists():
                        # Copy visualization to organized structure
                        viz_name = Path(viz_path).name
                        import shutil
                        shutil.copy2(viz_path, viz_dir / viz_name)
            
            # Save additional formats based on engine type
            self.save_engine_specific_formats(result, data_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving result formats: {e}")
    
    def save_result_as_csv(self, result: Dict[str, Any], csv_path: Path):
        """Save result as flattened CSV"""
        try:
            # Flatten nested dictionaries
            flattened = self.flatten_dict(result)
            
            # Create DataFrame
            df = pd.DataFrame([flattened])
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
    
    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def save_engine_specific_formats(self, result: Dict[str, Any], data_dir: Path):
        """Save engine-specific formats - to be overridden by subclasses"""
        pass
    
    def save_strategy_summary(self, strategy_results: List[Dict[str, Any]], strategy_dir: Path):
        """Save strategy summary"""
        try:
            if not strategy_results:
                return
            
            # Create summary
            summary = {
                'strategy_name': strategy_results[0].get('strategy_file', 'unknown'),
                'total_tests': len(strategy_results),
                'data_files_tested': len(set(r.get('data_file', '') for r in strategy_results)),
                'engine': strategy_results[0].get('engine', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'results': strategy_results
            }
            
            # Save summary JSON
            summary_path = strategy_dir / "strategy_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save summary CSV
            summary_csv_path = strategy_dir / "strategy_summary.csv"
            df = pd.DataFrame(strategy_results)
            df.to_csv(summary_csv_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving strategy summary: {e}")
