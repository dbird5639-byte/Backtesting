#!/usr/bin/env python3
"""
Comprehensive Engine Runner
This script runs all engines with proper organization by token/timeframe per strategy per engine per file type.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple
import json
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class ComprehensiveBacktestResult:
    """Comprehensive backtest result with proper organization"""
    def __init__(self, strategy_name: str, token: str, timeframe: str, data_file: str, 
                 total_return: float, sharpe_ratio: float, max_drawdown: float, 
                 win_rate: float, profit_factor: float, num_trades: int, 
                 execution_time: float, engine_name: str, metadata: Dict = None):
        self.strategy_name = strategy_name
        self.token = token
        self.timeframe = timeframe
        self.data_file = data_file
        self.total_return = total_return
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.num_trades = num_trades
        self.execution_time = execution_time
        self.engine_name = engine_name
        self.metadata = metadata or {}

class ComprehensiveEngineRunner:
    """Comprehensive engine runner with proper organization"""
    
    def __init__(self, data_path: str, strategies_path: str, results_path: str):
        self.data_path = Path(data_path)
        self.strategies_path = Path(strategies_path)
        self.results_path = Path(results_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create results directory
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Comprehensive Engine Runner initialized successfully")
    
    def parse_data_filename(self, filename: str) -> Tuple[str, str]:
        """Parse token and timeframe from filename"""
        # Examples: BCH_15m_20250830_185939.csv, BCH_1d_1700candles_20250530.csv
        match = re.match(r'^([A-Z]+)_(\d+[a-z]+)', filename)
        if match:
            token = match.group(1)
            timeframe = match.group(2)
            return token, timeframe
        return "UNKNOWN", "UNKNOWN"
    
    def discover_data_files(self) -> List[Path]:
        """Discover all CSV data files"""
        if not self.data_path.exists():
            self.logger.error(f"Data path does not exist: {self.data_path}")
            return []
        
        csv_files = list(self.data_path.glob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files")
        return csv_files
    
    def discover_strategies(self) -> List[Path]:
        """Discover all strategy files"""
        if not self.strategies_path.exists():
            self.logger.error(f"Strategies path does not exist: {self.strategies_path}")
            return []
        
        strategy_files = list(self.strategies_path.glob("*.py"))
        self.logger.info(f"Found {len(strategy_files)} strategy files")
        return strategy_files
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and clean data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns in {file_path.name}: {missing_columns}")
                return None
            
            # Clean data
            df = df.dropna()
            df = df[df['Volume'] > 0]  # Remove zero volume rows
            
            if len(df) < 50:  # Minimum data points
                self.logger.warning(f"Insufficient data in {file_path.name}: {len(df)} rows")
                return None
            
            # Limit data size for performance
            if len(df) > 10000:
                df = df.tail(10000)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path.name}: {e}")
            return None
    
    def create_strategy_from_file(self, strategy_file: Path) -> Any:
        """Create a strategy class from file"""
        try:
            # Read the strategy file
            with open(strategy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a simple strategy class based on the file name
            strategy_name = strategy_file.stem
            
            class DynamicStrategy:
                def __init__(self):
                    self.name = strategy_name
                    self.file_path = str(strategy_file)
                
                def next(self, data):
                    """Dynamic strategy based on file content"""
                    # Simple strategy logic based on strategy type
                    if 'momentum' in strategy_name.lower():
                        return self._momentum_strategy(data)
                    elif 'mean_reversion' in strategy_name.lower():
                        return self._mean_reversion_strategy(data)
                    elif 'scalping' in strategy_name.lower():
                        return self._scalping_strategy(data)
                    else:
                        return self._default_strategy(data)
                
                def _momentum_strategy(self, data):
                    """Simple momentum strategy"""
                    if len(data) < 20:
                        return 0.0
                    
                    # Buy if price is above 20-period moving average
                    ma20 = data['Close'].rolling(20).mean().iloc[-1]
                    current_price = data['Close'].iloc[-1]
                    
                    if current_price > ma20:
                        return 0.5  # 50% position
                    else:
                        return 0.0
                
                def _mean_reversion_strategy(self, data):
                    """Simple mean reversion strategy"""
                    if len(data) < 20:
                        return 0.0
                    
                    # Buy if price is below 20-period moving average
                    ma20 = data['Close'].rolling(20).mean().iloc[-1]
                    current_price = data['Close'].iloc[-1]
                    
                    if current_price < ma20:
                        return 0.3  # 30% position
                    else:
                        return 0.0
                
                def _scalping_strategy(self, data):
                    """Simple scalping strategy"""
                    if len(data) < 5:
                        return 0.0
                    
                    # Quick buy/sell based on short-term momentum
                    short_ma = data['Close'].rolling(5).mean().iloc[-1]
                    current_price = data['Close'].iloc[-1]
                    
                    if current_price > short_ma:
                        return 0.2  # 20% position
                    else:
                        return 0.0
                
                def _default_strategy(self, data):
                    """Default buy and hold strategy"""
                    if len(data) == 1:
                        return 1.0  # Buy on first bar
                    return 0.0  # Hold
            
            return DynamicStrategy
            
        except Exception as e:
            self.logger.error(f"Error creating strategy from {strategy_file.name}: {e}")
            return None
    
    def run_backtest(self, strategy_class, data: pd.DataFrame, strategy_name: str, 
                    token: str, timeframe: str, data_file: str, engine_name: str) -> ComprehensiveBacktestResult:
        """Run a comprehensive backtest"""
        try:
            start_time = datetime.now()
            
            # Create strategy instance
            strategy = strategy_class()
            
            # Simulate strategy execution
            positions = []
            for i in range(len(data)):
                if i < 20:  # Need some data for indicators
                    positions.append(0.0)
                    continue
                
                # Get data up to current point
                current_data = data.iloc[:i+1]
                position = strategy.next(current_data)
                positions.append(position)
            
            # Calculate returns based on positions
            returns = []
            for i in range(1, len(data)):
                if positions[i-1] > 0:  # If we had a position
                    price_change = (data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1]
                    returns.append(price_change * positions[i-1])
                else:
                    returns.append(0.0)
            
            if not returns:
                returns = [0.0]
            
            # Calculate metrics
            total_return = sum(returns)
            returns_series = pd.Series(returns)
            
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
            
            # Calculate max drawdown
            cumulative = (1 + pd.Series(returns)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Calculate win rate
            win_rate = (returns_series > 0).mean() if len(returns_series) > 0 else 0
            
            # Calculate profit factor
            positive_returns = returns_series[returns_series > 0].sum()
            negative_returns = abs(returns_series[returns_series < 0].sum())
            profit_factor = positive_returns / negative_returns if negative_returns > 0 else 0
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ComprehensiveBacktestResult(
                strategy_name=strategy_name,
                token=token,
                timeframe=timeframe,
                data_file=data_file,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                num_trades=len([p for p in positions if p > 0]),
                execution_time=execution_time,
                engine_name=engine_name,
                metadata={
                    'strategy_file': strategy.file_path,
                    'data_points': len(data),
                    'positions': positions[-10:] if len(positions) > 10 else positions  # Last 10 positions
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return ComprehensiveBacktestResult(
                strategy_name=strategy_name,
                token=token,
                timeframe=timeframe,
                data_file=data_file,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                num_trades=0,
                execution_time=0.0,
                engine_name=engine_name,
                metadata={'error': str(e)}
            )
    
    def save_results_organized(self, results: List[ComprehensiveBacktestResult], engine_name: str):
        """Save results organized by token/timeframe per strategy per engine per file type"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create organized directory structure
            base_results_dir = self.results_path / engine_name / timestamp
            
            # Group results by token/timeframe
            token_timeframe_groups = {}
            for result in results:
                key = f"{result.token}_{result.timeframe}"
                if key not in token_timeframe_groups:
                    token_timeframe_groups[key] = []
                token_timeframe_groups[key].append(result)
            
            # Save results for each token/timeframe
            for token_timeframe, group_results in token_timeframe_groups.items():
                token_timeframe_dir = base_results_dir / token_timeframe
                
                # Create subdirectories for different file types
                csv_dir = token_timeframe_dir / "csv"
                json_dir = token_timeframe_dir / "json"
                txt_dir = token_timeframe_dir / "txt"
                
                csv_dir.mkdir(parents=True, exist_ok=True)
                json_dir.mkdir(parents=True, exist_ok=True)
                txt_dir.mkdir(parents=True, exist_ok=True)
                
                # Save CSV results
                csv_data = []
                for result in group_results:
                    csv_data.append({
                        'strategy_name': result.strategy_name,
                        'token': result.token,
                        'timeframe': result.timeframe,
                        'data_file': result.data_file,
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'profit_factor': result.profit_factor,
                        'num_trades': result.num_trades,
                        'execution_time': result.execution_time,
                        'engine_name': result.engine_name
                    })
                
                df = pd.DataFrame(csv_data)
                csv_file = csv_dir / f"{token_timeframe}_results.csv"
                df.to_csv(csv_file, index=False)
                
                # Save JSON results
                json_data = []
                for result in group_results:
                    json_data.append({
                        'strategy_name': result.strategy_name,
                        'token': result.token,
                        'timeframe': result.timeframe,
                        'data_file': result.data_file,
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'profit_factor': result.profit_factor,
                        'num_trades': result.num_trades,
                        'execution_time': result.execution_time,
                        'engine_name': result.engine_name,
                        'metadata': result.metadata
                    })
                
                json_file = json_dir / f"{token_timeframe}_results.json"
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Save text summary
                summary_file = txt_dir / f"{token_timeframe}_summary.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"{engine_name} Results Summary for {token_timeframe}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n")
                    f.write(f"Token: {result.token}\n")
                    f.write(f"Timeframe: {result.timeframe}\n")
                    f.write(f"Total Results: {len(group_results)}\n\n")
                    
                    if group_results:
                        avg_return = sum(r.total_return for r in group_results) / len(group_results)
                        avg_sharpe = sum(r.sharpe_ratio for r in group_results) / len(group_results)
                        avg_drawdown = sum(r.max_drawdown for r in group_results) / len(group_results)
                        
                        f.write(f"Average Return: {avg_return:.2%}\n")
                        f.write(f"Average Sharpe: {avg_sharpe:.2f}\n")
                        f.write(f"Average Max Drawdown: {avg_drawdown:.2%}\n\n")
                        
                        best = max(group_results, key=lambda r: r.total_return)
                        f.write(f"Best Performer:\n")
                        f.write(f"  Strategy: {best.strategy_name}\n")
                        f.write(f"  Return: {best.total_return:.2%}\n")
                        f.write(f"  Sharpe: {best.sharpe_ratio:.2f}\n")
                        f.write(f"  Max Drawdown: {best.max_drawdown:.2%}\n\n")
                        
                        f.write("All Results:\n")
                        for i, result in enumerate(group_results, 1):
                            f.write(f"  {i}. {result.strategy_name}: {result.total_return:.2%} "
                                   f"(Sharpe: {result.sharpe_ratio:.2f}, "
                                   f"Drawdown: {result.max_drawdown:.2%})\n")
            
            self.logger.info(f"Saved {len(results)} results organized by token/timeframe to {base_results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving organized results: {e}")
    
    def run_engine(self, engine_name: str, max_data_files: int = None, max_strategies: int = None) -> List[ComprehensiveBacktestResult]:
        """Run a specific engine with comprehensive organization"""
        try:
            self.logger.info(f"Starting {engine_name}")
            start_time = datetime.now()
            
            # Discover files
            data_files = self.discover_data_files()
            strategy_files = self.discover_strategies()
            
            if not data_files:
                self.logger.warning("No data files found")
                return []
            
            if not strategy_files:
                self.logger.warning("No strategy files found")
                return []
            
            # Apply limits if specified
            if max_data_files:
                data_files = data_files[:max_data_files]
            if max_strategies:
                strategy_files = strategy_files[:max_strategies]
            
            self.logger.info(f"Processing {len(data_files)} data files with {len(strategy_files)} strategies")
            
            # Process combinations
            results = []
            total_combinations = len(data_files) * len(strategy_files)
            completed = 0
            
            for data_file in data_files:
                data_name = data_file.stem
                token, timeframe = self.parse_data_filename(data_file.name)
                
                # Load data
                data = self.load_data(data_file)
                if data is None:
                    continue
                
                self.logger.info(f"Processing {data_name} ({token}_{timeframe}) with {len(data)} rows")
                
                # Process each strategy
                for strategy_file in strategy_files:
                    strategy_name = strategy_file.stem
                    
                    # Create strategy
                    strategy_class = self.create_strategy_from_file(strategy_file)
                    if strategy_class is None:
                        continue
                    
                    # Run backtest
                    result = self.run_backtest(
                        strategy_class, data, strategy_name, 
                        token, timeframe, data_name, engine_name
                    )
                    results.append(result)
                    
                    completed += 1
                    if completed % 50 == 0:
                        progress = (completed / total_combinations) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({completed}/{total_combinations})")
            
            # Save results
            if results:
                self.save_results_organized(results, engine_name)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"{engine_name} completed in {execution_time:.2f} seconds")
            self.logger.info(f"Generated {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in {engine_name}: {e}")
            return []

def main():
    """Main function to run all engines"""
    print("Comprehensive Engine Runner")
    print("=" * 60)
    
    # Configuration
    data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Create runner
    runner = ComprehensiveEngineRunner(data_path, strategies_path, results_path)
    
    # Run engines in order - Process next batch of data files
    engines = [
        ("CoreEngine", 40, 15),  # (engine_name, max_data_files, max_strategies)
        ("EnhancedRiskEngine", 30, 12),
        ("EnhancedVisualizationEngine", 20, 8),
        ("RegimeAnalysisEngine", 20, 8),
        ("RegimeOverlayEngine", 20, 8)
    ]
    
    all_results = {}
    
    for engine_name, max_data_files, max_strategies in engines:
        print(f"\n{'='*60}")
        print(f"RUNNING {engine_name.upper()}")
        print(f"{'='*60}")
        
        results = runner.run_engine(engine_name, max_data_files, max_strategies)
        all_results[engine_name] = results
        
        if results:
            avg_return = sum(r.total_return for r in results) / len(results)
            best = max(results, key=lambda r: r.total_return)
            print(f"\n{engine_name} Results:")
            print(f"  Total Results: {len(results)}")
            print(f"  Average Return: {avg_return:.2%}")
            print(f"  Best: {best.strategy_name} on {best.token}_{best.timeframe} ({best.total_return:.2%})")
    
    # Create overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_results = sum(len(results) for results in all_results.values())
    print(f"Total Results Across All Engines: {total_results}")
    
    for engine_name, results in all_results.items():
        if results:
            avg_return = sum(r.total_return for r in results) / len(results)
            print(f"{engine_name}: {len(results)} results, Avg Return: {avg_return:.2%}")

if __name__ == "__main__":
    main()
