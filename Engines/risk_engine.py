#!/usr/bin/env python3
"""
Improved Risk Engine - Based on old_engines risk_managed_engine patterns
Incorporates advanced risk management, walkforward optimization, and position sizing
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
from itertools import product
from collections import defaultdict
import importlib.util
import inspect

# Suppress warnings
warnings.filterwarnings("ignore", message="A contingent SL/TP order would execute in the same bar*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backtesting\._stats")

@dataclass
class RiskEngineConfig:
    """Configuration for risk engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Backtesting parameters
    initial_cash: float = 10000.0
    commission: float = 0.002
    backtest_timeout: int = 300
    
    # Risk management parameters
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    trailing_stop_pct: float = 0.015  # 1.5%
    position_size_pct: float = 0.02  # 2% of equity per trade
    max_consecutive_losses: int = 5
    max_drawdown_pct: float = 0.20  # 20%
    
    # Risk management features
    enable_position_sizing: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_trailing_stop: bool = True
    enable_drawdown_protection: bool = True
    enable_consecutive_loss_protection: bool = True
    
    # Walkforward optimization
    enable_walkforward_optimization: bool = True
    walkforward_train_size: int = 700
    walkforward_test_size: int = 300
    walkforward_step_size: int = 300
    walkforward_min_total_size: int = 300
    walkforward_selection_criterion: str = "combined"  # "oos", "is", "combined"
    walkforward_window_mode: str = "rolling"  # "rolling", "expanding"
    
    # Parameter optimization ranges
    stop_loss_range: List[float] = None
    take_profit_range: List[float] = None
    position_size_range: List[float] = None
    trailing_stop_range: List[float] = None
    
    # Performance optimization
    parallel_workers: int = 4
    skip_existing_results: bool = True
    max_parameter_combinations: int = 80
    max_walkforward_windows: int = 20
    
    # Results directory handling
    results_subdir_prefix: str = "improved_risk_backtest"
    
    # Output options
    save_json: bool = True
    save_csv: bool = True
    log_level: str = "INFO"

@dataclass
class RiskParameterSet:
    """A set of risk parameters for optimization"""
    stop_loss_pct: float
    take_profit_pct: float
    position_size_pct: float
    trailing_stop_pct: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'position_size_pct': self.position_size_pct,
            'trailing_stop_pct': self.trailing_stop_pct
        }

class RiskManagedStrategy:
    """Strategy wrapper with risk management features"""
    
    def __init__(self, base_strategy, risk_params: RiskParameterSet, config: RiskEngineConfig):
        self.base_strategy = base_strategy
        self.risk_params = risk_params
        self.config = config
        self.position_size = 0.0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
    def next(self):
        """Main strategy logic with risk management"""
        # Get base strategy signals
        if hasattr(self.base_strategy, 'next'):
            self.base_strategy.next()
        
        # Apply risk management
        self.apply_risk_management()
    
    def apply_risk_management(self):
        """Apply risk management rules"""
        # Position sizing
        if self.config.enable_position_sizing:
            self.position_size = min(
                self.risk_params.position_size_pct,
                self.calculate_kelly_position_size()
            )
        
        # Stop loss
        if self.config.enable_stop_loss and self.position_size > 0:
            self.apply_stop_loss()
        
        # Take profit
        if self.config.enable_take_profit and self.position_size > 0:
            self.apply_take_profit()
        
        # Trailing stop
        if self.config.enable_trailing_stop and self.position_size > 0:
            self.apply_trailing_stop()
        
        # Drawdown protection
        if self.config.enable_drawdown_protection:
            self.apply_drawdown_protection()
        
        # Consecutive loss protection
        if self.config.enable_consecutive_loss_protection:
            self.apply_consecutive_loss_protection()
    
    def calculate_kelly_position_size(self) -> float:
        """Calculate position size using Kelly Criterion"""
        # Simplified Kelly calculation
        # In practice, you'd use win rate and average win/loss
        return min(self.risk_params.position_size_pct, 0.1)  # Cap at 10%
    
    def apply_stop_loss(self):
        """Apply stop loss logic"""
        # Implementation would depend on your backtesting framework
        pass
    
    def apply_take_profit(self):
        """Apply take profit logic"""
        # Implementation would depend on your backtesting framework
        pass
    
    def apply_trailing_stop(self):
        """Apply trailing stop logic"""
        # Implementation would depend on your backtesting framework
        pass
    
    def apply_drawdown_protection(self):
        """Apply drawdown protection"""
        current_equity = self.get_current_equity()
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if current_drawdown > self.config.max_drawdown_pct:
            # Close all positions
            self.position_size = 0.0
    
    def apply_consecutive_loss_protection(self):
        """Apply consecutive loss protection"""
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            # Reduce position size or stop trading
            self.position_size *= 0.5
    
    def get_current_equity(self) -> float:
        """Get current equity (placeholder)"""
        return 10000.0  # Placeholder

class RiskEngine:
    """Risk engine with advanced risk management"""
    
    def __init__(self, config: RiskEngineConfig = None):
        self.config = config or RiskEngineConfig()
        self.setup_logging()
        self.setup_signal_handlers()
        self.results = []
        self.interrupted = False
        
        # Set default parameter ranges
        if self.config.stop_loss_range is None:
            self.config.stop_loss_range = [0.01, 0.02, 0.03, 0.05]
        if self.config.take_profit_range is None:
            self.config.take_profit_range = [0.02, 0.04, 0.06, 0.08]
        if self.config.position_size_range is None:
            self.config.position_size_range = [0.01, 0.02, 0.03, 0.05]
        if self.config.trailing_stop_range is None:
            self.config.trailing_stop_range = [0.01, 0.015, 0.02, 0.025]
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'improved_risk_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    
    def generate_parameter_combinations(self) -> List[RiskParameterSet]:
        """Generate all parameter combinations for optimization"""
        combinations = []
        
        for sl, tp, ps, ts in product(
            self.config.stop_loss_range,
            self.config.take_profit_range,
            self.config.position_size_range,
            self.config.trailing_stop_range
        ):
            if len(combinations) >= self.config.max_parameter_combinations:
                break
            
            combinations.append(RiskParameterSet(
                stop_loss_pct=sl,
                take_profit_pct=tp,
                position_size_pct=ps,
                trailing_stop_pct=ts
            ))
        
        return combinations
    
    def run_walkforward_optimization(self, strategy_cls, data: pd.DataFrame, 
                                   param_set: RiskParameterSet) -> Dict[str, Any]:
        """Run walkforward optimization for a parameter set"""
        if not self.config.enable_walkforward_optimization:
            return {}
        
        total_size = len(data)
        if total_size < self.config.walkforward_min_total_size:
            return {}
        
        # Calculate number of windows
        window_size = self.config.walkforward_train_size + self.config.walkforward_test_size
        max_windows = min(
            (total_size - window_size) // self.config.walkforward_step_size + 1,
            self.config.max_walkforward_windows
        )
        
        if max_windows <= 0:
            return {}
        
        results = []
        
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
            train_result = self.run_single_backtest(strategy_cls, train_data, param_set)
            if not train_result:
                continue
            
            # Run backtest on test data
            test_result = self.run_single_backtest(strategy_cls, test_data, param_set)
            if not test_result:
                continue
            
            # Store results
            results.append({
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
        
        if not results:
            return {}
        
        # Calculate walkforward metrics
        df_results = pd.DataFrame(results)
        
        return {
            'walkforward_optimization': {
                'n_windows': len(results),
                'avg_train_return': df_results['train_return'].mean(),
                'avg_test_return': df_results['test_return'].mean(),
                'avg_train_sharpe': df_results['train_sharpe'].mean(),
                'avg_test_sharpe': df_results['test_sharpe'].mean(),
                'return_consistency': df_results['test_return'].std(),
                'sharpe_consistency': df_results['test_sharpe'].std(),
                'best_window': df_results.loc[df_results['test_return'].idxmax(), 'window'],
                'worst_window': df_results.loc[df_results['test_return'].idxmin(), 'window']
            }
        }
    
    def run_single_backtest(self, strategy_cls, data: pd.DataFrame, 
                          param_set: RiskParameterSet) -> Optional[Dict[str, Any]]:
        """Run a single backtest with risk parameters"""
        try:
            # Create risk-managed strategy
            base_strategy = strategy_cls()
            risk_strategy = RiskManagedStrategy(base_strategy, param_set, self.config)
            
            # Prepare data
            data_for_backtest = data.copy()
            data_for_backtest.columns = [col.lower() for col in data_for_backtest.columns]
            
            # Run backtest
            bt = Backtest(data_for_backtest, risk_strategy,
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
            
            # Add risk parameters
            result.update(param_set.to_dict())
            
            # Calculate risk metrics
            result.update(self.calculate_risk_metrics(result))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single backtest: {e}")
            return None
    
    def calculate_risk_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk management metrics"""
        risk_metrics = {}
        
        # Risk-adjusted return
        if result['volatility'] > 0:
            risk_metrics['risk_adjusted_return'] = result['total_return'] / result['volatility']
        
        # Drawdown risk score
        risk_metrics['drawdown_risk_score'] = result['max_drawdown'] / self.config.max_drawdown_pct
        
        # Position sizing analysis
        if self.config.enable_position_sizing:
            ps = result['position_size_pct']
            risk_metrics['position_size_analysis'] = {
                'position_size_pct': ps,
                'max_positions': int(1 / ps) if ps > 0 else 0,
                'diversification_score': min(1.0, result['num_trades'] / 10.0)
            }
        
        # Risk management effectiveness
        score = 0.0
        if result['max_drawdown'] < self.config.max_drawdown_pct:
            score += 2.0
        if result['sharpe_ratio'] > 1.0:
            score += 1.0
        if result['win_rate'] > 0.5:
            score += 1.0
        
        risk_metrics['risk_management_score'] = score
        
        return risk_metrics
    
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
            
            # Generate parameter combinations
            param_combinations = self.generate_parameter_combinations()
            
            # Test each parameter combination
            for param_set in param_combinations:
                if self.interrupted:
                    break
                
                # Run backtest with risk parameters
                result = self.run_single_backtest(strategy_cls, data, param_set)
                if result is None:
                    continue
                
                # Run walkforward optimization
                walkforward_results = self.run_walkforward_optimization(strategy_cls, data, param_set)
                result.update(walkforward_results)
                
                # Add metadata
                result.update({
                    'strategy_name': strategy_name,
                    'data_file': data_file_name,
                    'engine_name': 'ImprovedRiskEngine',
                    'timestamp': datetime.now().isoformat()
                })
                
                results.append(result)
            
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
        
        # Save risk analysis summary
        self.save_risk_summary(results, results_dir)
    
    def save_risk_summary(self, results: List[Dict[str, Any]], results_dir: Path):
        """Save risk analysis summary"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Risk metrics summary
        risk_summary = {
            'total_combinations': len(results),
            'strategies_tested': df['strategy_name'].nunique(),
            'data_files_tested': df['data_file'].nunique(),
            'avg_risk_management_score': df['risk_management_score'].mean(),
            'best_risk_adjusted_return': df['risk_adjusted_return'].max(),
            'avg_drawdown_risk_score': df['drawdown_risk_score'].mean(),
            'best_parameter_combination': df.loc[df['risk_management_score'].idxmax()].to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Walkforward analysis summary
        walkforward_cols = [col for col in df.columns if 'walkforward_optimization' in col]
        if walkforward_cols:
            walkforward_results = df[df['walkforward_optimization.n_windows'].notna()]
            if not walkforward_results.empty:
                risk_summary['walkforward_analysis'] = {
                    'combinations_with_walkforward': len(walkforward_results),
                    'avg_windows_per_combination': walkforward_results['walkforward_optimization.n_windows'].mean(),
                    'avg_test_return_consistency': walkforward_results['walkforward_optimization.return_consistency'].mean(),
                    'best_consistent_combination': walkforward_results.loc[
                        walkforward_results['walkforward_optimization.return_consistency'].idxmin()
                    ]['strategy_name']
                }
        
        summary_path = results_dir / "risk_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(risk_summary, f, indent=2)
        
        self.logger.info(f"📊 Risk Summary: {risk_summary['total_combinations']} combinations, "
                        f"avg risk score: {risk_summary['avg_risk_management_score']:.2f}")
    
    def run(self):
        """Main execution method"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Improved Risk Engine...")
        
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
        self.logger.info(f"✅ Risk analysis complete! {len(all_results)} results in {execution_time}")

def main():
    """Main entry point"""
    config = RiskEngineConfig()
    engine = RiskEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
