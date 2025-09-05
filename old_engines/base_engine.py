"""
Base Engine for Backtesting

This module provides the base class for all backtesting engines with common
functionality like data loading, strategy loading, and result handling.
"""

import os
import importlib.util
import inspect
import logging
import time
import pandas as pd
import numpy as np
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from backtesting import Backtest
import warnings

"""Centralized warning controls to keep terminal output clean."""
# Suppress known warnings
warnings.filterwarnings(
    "ignore",
    message="A contingent SL/TP order would execute in the same bar*",
)
# Suppress noisy numpy RuntimeWarnings emitted inside backtesting._stats (e.g., Sortino divide-by-zero)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"backtesting\._stats",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in scalar divide",
    category=RuntimeWarning,
)


class _ConsecutiveDuplicateFilter(logging.Filter):
    """Filter out consecutive duplicate log messages on the same level.

    Keeps file logs intact (we'll attach this only to console handler).
    """

    def __init__(self):
        super().__init__()
        self._last_key = None

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        key = (record.levelno, record.getMessage())
        if key == self._last_key:
            return False
        self._last_key = key
        return True

@dataclass
class EngineConfig:
    """Base configuration for all engines"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\Data\winners"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\results"
    
    # Backtest parameters
    initial_cash: float = 100000.0
    commission: float = 0.002
    backtest_timeout: int = 43200
    # Threading
    backtest_thread_workers: int = 4
    
    # Data processing
    min_data_points: int = 5
    max_data_points: int = 50000
    truncate_excess_data: bool = True
    truncate_side: str = "tail"  # "head" or "tail"
    
    # Output options
    save_json: bool = True
    save_csv: bool = False
    save_plots: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    skip_existing_results: bool = True # New: Add this line

class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for pandas and numpy objects"""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return str(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Gracefully handle class and function objects by name to avoid "not JSON serializable"
        try:
            import inspect as _inspect  # local import to avoid top-level dependency
            if _inspect.isclass(obj):
                return getattr(obj, '__name__', str(obj))
            if _inspect.isfunction(obj):
                return getattr(obj, '__name__', str(obj))
        except Exception:
            pass
        return super().default(obj)

class BaseEngine:
    """Base class for all backtesting engines"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.shutdown_requested = False
        self._setup_signal_handlers()
        # Reusable executor to avoid creating thousands of threads inside loops
        # This directly addresses "can't start new thread" under heavy walkforward/param sweeps
        self._backtest_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max(1, int(getattr(self.config, 'backtest_thread_workers', 4)))
        )
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler (with duplicate suppression)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.addFilter(_ConsecutiveDuplicateFilter())
            logger.addHandler(console_handler)
            
            # File handler if enabled
            if self.config.log_to_file:
                log_dir = Path(self.config.results_path) / "logs"
                log_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"{self.__class__.__name__.lower()}_{timestamp}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info("Interrupt signal received. Shutting down gracefully...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for path in [self.config.data_path, self.config.strategies_path, self.config.results_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def discover_data_files(self) -> List[str]:
        """Discover all CSV data files recursively"""
        data_path = Path(self.config.data_path)
        csv_files = list(data_path.glob("**/*.csv"))
        return [str(f) for f in csv_files]
    
    def get_strategy_name(self, strategy_path: str) -> str:
        """Return strategy name (filename without extension) from a path"""
        return os.path.splitext(os.path.basename(strategy_path))[0]
    
    def discover_data_files_for_strategy(self, strategy_path: str) -> List[str]:
        """Discover CSV data files for a specific strategy under winners/<strategy_name>.

        Falls back to all data under `data_path` if the strategy-specific folder
        doesn't exist or is empty.
        """
        try:
            winners_root = Path(self.config.data_path)
            strategy_name = self.get_strategy_name(strategy_path)
            candidate_dir = winners_root / strategy_name
            if candidate_dir.exists() and candidate_dir.is_dir():
                csv_files = list(candidate_dir.glob("**/*.csv"))
                if csv_files:
                    return [str(f) for f in csv_files]
            # Fallback to all data under data_path
            return self.discover_data_files()
        except Exception:
            # Defensive fallback
            return self.discover_data_files()
    
    def discover_strategies(self) -> List[str]:
        """Discover all strategy files"""
        strategies_path = Path(self.config.strategies_path)
        py_files = list(strategies_path.glob("**/*.py"))
        # Filter out __init__.py and other non-strategy files
        strategy_files = [f for f in py_files if not f.name.startswith('__')]
        return [str(f) for f in strategy_files]
    
    def load_strategy(self, strategy_path: str) -> type:
        """Dynamically load strategy class from file path"""
        try:
            spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
            if spec is None:
                raise ValueError(f"Could not load spec for {strategy_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            
            # Find the strategy class (first class that inherits from Strategy)
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, '__bases__') and 
                    any('Strategy' in base.__name__ for base in obj.__bases__)):
                    return obj
            
            raise ValueError(f"No strategy class found in {strategy_path}")
        except Exception as e:
            self.logger.error(f"Failed to load strategy from {strategy_path}: {e}")
            raise
    
    def load_and_validate_data(self, data_file: str) -> pd.DataFrame:
        """Load and validate data from CSV file"""
        try:
            df = pd.read_csv(data_file)
            df.columns = df.columns.str.strip()
            
            # Find date column
            date_col = None
            for col in df.columns:
                if col.lower() in ['date', 'time', 'timestamp', 'datetime']:
                    date_col = col
                    break
            
            if date_col is None:
                raise ValueError(f"No date column found in {data_file}")
            
            # Set date as index and handle duplicates
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Handle duplicate index labels
            if df.index.duplicated().any():
                self.logger.warning(f"Duplicate timestamps found in {data_file}, keeping first occurrence")
                df = df[~df.index.duplicated(keep='first')]
            
            # Standardize column names
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            }
            df = df.rename(columns={col: column_mapping[col.lower()] 
                                  for col in df.columns 
                                  if col.lower() in column_mapping})
            
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate data size (and optionally truncate instead of failing)
            if len(df) < self.config.min_data_points:
                raise ValueError(f"Insufficient data points: {len(df)} < {self.config.min_data_points}")
            
            if len(df) > self.config.max_data_points:
                if getattr(self.config, 'truncate_excess_data', True):
                    original_len = len(df)
                    if getattr(self.config, 'truncate_side', 'tail') == 'head':
                        df = df.iloc[: self.config.max_data_points]
                    else:
                        df = df.iloc[-self.config.max_data_points :]
                    self.logger.info(
                        f"Truncated data from {original_len} to {len(df)} rows for {data_file}"
                    )
                else:
                    raise ValueError(f"Too many data points: {len(df)} > {self.config.max_data_points}")
            
            # Drop non-finite values in OHLCV only, forward-fill last observation for gaps
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].ffill()
            df = df.dropna(subset=['Open','High','Low','Close','Volume'])
            
            if len(df) == 0:
                raise ValueError("No valid data after dropping NaN values")
            
            # Ensure numeric dtypes for OHLCV
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows made NaN by coercion and enforce finiteness
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].ffill()
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

            # Validate OHLC relationships
            invalid_ohlc = (
                (df['High'] < df['Low']) | 
                (df['Open'] > df['High']) | 
                (df['Open'] < df['Low']) |
                (df['Close'] > df['High']) | 
                (df['Close'] < df['Low'])
            )
            if invalid_ohlc.any():
                raise ValueError(f"Invalid OHLC relationships found in {data_file}")
            
            # Check for zero or negative prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    raise ValueError(f"Zero or negative prices found in {col} column")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {data_file}: {e}")
            raise
    
    def extract_info_from_filename(self, filename: str) -> Tuple[str, str]:
        """Extract symbol and timeframe from filename"""
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Extract symbol (first part before underscore)
        parts = name_without_ext.split('_')
        symbol = parts[0] if len(parts) >= 1 else "UNKNOWN"
        
        # Extract timeframe (look for patterns like 1m, 5m, 15m, 1h, 4h, 1d)
        timeframe = "UNKNOWN"
        for part in parts:
            if any(tf in part.lower() for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']):
                timeframe = part
                break
        
        return symbol, timeframe
    
    def run_backtest(self, df: pd.DataFrame, strategy_cls: type) -> Optional[Dict[str, Any]]:
        """Run a backtest with timeout and error handling"""
        try:
            bt = Backtest(df, strategy_cls, 
                         cash=self.config.initial_cash, 
                         commission=self.config.commission)
            # Submit to the shared executor rather than creating a new one each call
            future = self._backtest_executor.submit(bt.run)
            stats = future.result(timeout=self.config.backtest_timeout)
            
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
            
        except FuturesTimeoutError:
            self.logger.error("Backtest timeout")
            return None
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return None

    def __del__(self):
        try:
            if hasattr(self, '_backtest_executor') and self._backtest_executor:
                self._backtest_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    
    def assess_quality(self, stats: Union[dict, pd.Series]) -> Dict[str, Any]:
        """Assess strategy quality based on multiple metrics"""
        quality_score = 0
        max_score = 5
        
        def safe_get(stat_name: str, default: float = 0.0) -> float:
            if isinstance(stats, dict):
                value = stats.get(stat_name, default)
            else:  # pd.Series
                value = stats[stat_name] if stat_name in stats.index else default
            if pd.isna(value):
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Sharpe ratio assessment
        sharpe = safe_get('Sharpe Ratio', 0.0)
        if sharpe > 1.5:
            quality_score += 1
        elif sharpe > 1.0:
            quality_score += 0.5
        
        # Profit factor assessment
        profit_factor = safe_get('Profit Factor', 0.0)
        if profit_factor > 2.0:
            quality_score += 1
        elif profit_factor > 1.5:
            quality_score += 0.5
        
        # Win rate assessment
        win_rate = safe_get('Win Rate [%]', 0.0)
        if win_rate > 60:
            quality_score += 1
        elif win_rate > 50:
            quality_score += 0.5
        
        # Drawdown assessment
        drawdown = abs(safe_get('Max. Drawdown [%]', 0.0))
        if drawdown < 15:
            quality_score += 1
        elif drawdown < 20:
            quality_score += 0.5
        
        # Return assessment
        return_pct = safe_get('Return [%]', 0.0)
        if return_pct > 20:
            quality_score += 1
        elif return_pct > 10:
            quality_score += 0.5
        
        quality_level = 'Excellent' if quality_score >= 4 else \
                       'Good' if quality_score >= 3 else \
                       'Fair' if quality_score >= 2 else 'Poor'
        
        return {
            'score': quality_score,
            'max_score': max_score,
            'level': quality_level,
            'percentage': (quality_score / max_score) * 100
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON and optionally CSV"""
        if self.config.save_json:
            json_path = f"{output_path}.json"
            # Atomic write to prevent truncated/corrupt files
            tmp_path = f"{json_path}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(results, f, indent=2, cls=CustomJsonEncoder)
            os.replace(tmp_path, json_path)
        
        if self.config.save_csv:
            csv_path = f"{output_path}.csv"
            pd.DataFrame([results]).to_csv(csv_path, index=False)

    def select_core_metrics(self, stats: Union[dict, pd.Series]) -> Dict[str, Any]:
        """Return a JSON-safe subset of core metrics from a stats dict/Series."""
        keys = [
            '# Trades', 'Win Rate [%]', 'Profit Factor', 'Sharpe Ratio',
            'Max. Drawdown [%]', 'Return [%]', 'Volatility (Ann.) [%]',
            'Calmar Ratio', 'Sortino Ratio', 'SQN', 'Equity Final [$]',
            'Equity Peak [$]', 'Buy & Hold Return [%]', 'Exposure Time [%]'
        ]
        core: Dict[str, Any] = {}
        def get_value(k: str) -> Any:
            if isinstance(stats, dict):
                return stats.get(k)
            return stats[k] if k in stats.index else None
        for k in keys:
            v = get_value(k)
            if v is None:
                continue
            try:
                # Ensure JSON serializable via custom encoder
                json.dumps(v, cls=CustomJsonEncoder)  # type: ignore[arg-type]
                core[k] = v
            except Exception:
                continue
        return core
    
    def create_results_directory(self, engine_name: str) -> str:
        """Create and return results directory for this engine run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_path) / f"{engine_name}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        return str(results_dir)
    
    def check_existing_result(self, strat_result_dir: str, data_name: str) -> bool:
        """Check if a result file already exists for this data file"""
        if not self.config.skip_existing_results:
            return False
        
        json_path = os.path.join(strat_result_dir, f"{data_name}.json")
        return os.path.exists(json_path)
    
    def get_existing_results_summary(self, results_dir: str) -> Dict[str, Any]:
        """Get summary of existing results to help with resuming"""
        summary = {
            'total_strategies': 0,
            'total_results': 0,
            'strategies_processed': [],
            'data_files_processed': [],
            'last_processed': None
        }
        
        if not os.path.exists(results_dir):
            return summary
        
        # Scan for existing results
        for strategy_dir in os.listdir(results_dir):
            strategy_path = os.path.join(results_dir, strategy_dir)
            if os.path.isdir(strategy_path):
                summary['total_strategies'] += 1
                strategy_results = []
                
                for file in os.listdir(strategy_path):
                    if file.endswith('.json'):
                        data_name = file.replace('.json', '')
                        strategy_results.append(data_name)
                        summary['data_files_processed'].append(f"{strategy_dir}/{data_name}")
                
                summary['strategies_processed'].append({
                    'strategy': strategy_dir,
                    'results_count': len(strategy_results),
                    'data_files': strategy_results
                })
                summary['total_results'] += len(strategy_results)
        
        return summary
    
    def find_last_processed_file(self, results_dir: str) -> Optional[Tuple[str, str]]:
        """Find the last processed strategy and data file based on file timestamps"""
        if not os.path.exists(results_dir):
            return None
        
        last_modified = 0
        last_strategy = None
        last_data_file = None
        
        for strategy_dir in os.listdir(results_dir):
            strategy_path = os.path.join(results_dir, strategy_dir)
            if os.path.isdir(strategy_path):
                for file in os.listdir(strategy_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(strategy_path, file)
                        file_time = os.path.getmtime(file_path)
                        if file_time > last_modified:
                            last_modified = file_time
                            last_strategy = strategy_dir
                            last_data_file = file.replace('.json', '')
        
        if last_strategy and last_data_file:
            return (last_strategy, last_data_file)
        return None
    
    def get_resume_info(self, results_dir: str) -> Dict[str, Any]:
        """Get information about where to resume from"""
        summary = self.get_existing_results_summary(results_dir)
        last_processed = self.find_last_processed_file(results_dir)
        
        resume_info = {
            'has_existing_results': summary['total_results'] > 0,
            'total_strategies_processed': summary['total_strategies'],
            'total_results_processed': summary['total_results'],
            'last_processed_strategy': last_processed[0] if last_processed else None,
            'last_processed_data_file': last_processed[1] if last_processed else None,
            'strategies_processed': summary['strategies_processed']
        }
        
        return resume_info
    
    def run(self):
        """Main run method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement the run method") 