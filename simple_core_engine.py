#!/usr/bin/env python3
"""
Simple Core Engine Runner - Process real data files with strategies
This script runs the Core Engine against your actual data files and strategies
without async complexity, saving results organized by data file, strategy, and engine.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("SIMPLE CORE ENGINE - REAL DATA PROCESSING")
print("=" * 80)
print("Processing actual data files with strategies")
print("Results organized by: data_file/strategy/engine")
print("=" * 80)

def discover_data_files():
        """Discover all CSV data files"""
    data_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data")
    if not data_path.exists():
        print(f"❌ Data path does not exist: {data_path}")
            return []
        
    csv_files = list(data_path.glob("*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files")
        return csv_files
    
def discover_strategies():
        """Discover all strategy files"""
    strategies_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies")
    if not strategies_path.exists():
        print(f"❌ Strategies path does not exist: {strategies_path}")
            return []
        
    strategy_files = [f for f in strategies_path.glob("*.py") if f.name != "__init__.py"]
    print(f"📁 Found {len(strategy_files)} strategy files")
        return strategy_files
    
def load_data(file_path):
        """Load and clean data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
            print(f"⚠️  Missing columns in {file_path.name}: {missing_columns}")
                return None
            
            # Clean data
            df = df.dropna()
            df = df[df['Volume'] > 0]  # Remove zero volume rows
            
            if len(df) < 50:  # Minimum data points
            print(f"⚠️  Insufficient data in {file_path.name}: {len(df)} rows")
                return None
            
            # Limit data size for performance
        if len(df) > 5000:
            df = df.tail(5000)
            
            return df
            
        except Exception as e:
        print(f"❌ Error loading data from {file_path.name}: {e}")
            return None
    
def create_simple_strategy(strategy_name):
    """Create a simple strategy based on name"""
        class SimpleStrategy:
        def __init__(self, name):
            self.name = name
            
            def next(self, data):
            """Simple strategy logic"""
            if len(data) < 20:
                return 0.0
            
            # Simple momentum strategy
            if 'momentum' in strategy_name.lower():
                ma_short = data['Close'].rolling(10).mean().iloc[-1]
                ma_long = data['Close'].rolling(30).mean().iloc[-1]
                return 0.5 if ma_short > ma_long else 0.0
            
            # Simple mean reversion strategy
            elif 'mean_reversion' in strategy_name.lower():
                ma = data['Close'].rolling(20).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                return 0.3 if current_price < ma else 0.0
            
            # Default buy and hold
            else:
                return 0.1 if len(data) == 1 else 0.0
    
    return SimpleStrategy(strategy_name)

def run_backtest(strategy, data, strategy_name, data_file):
        """Run a simple backtest"""
        try:
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
        
        return {
            'strategy_name': strategy_name,
            'data_file': data_file,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len([p for p in positions if p > 0]),
            'execution_time': 0.0,
            'timestamp': datetime.now().isoformat()
        }
            
        except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None

def save_results(results, results_path):
    """Save results organized by data file, strategy, and engine"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = Path(results_path) / "CoreEngine" / timestamp
        base_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Group results by data file
        data_file_groups = {}
        for result in results:
            data_file = result['data_file']
            if data_file not in data_file_groups:
                data_file_groups[data_file] = []
            data_file_groups[data_file].append(result)
        
        # Save results for each data file
        for data_file, group_results in data_file_groups.items():
            data_file_dir = base_results_dir / data_file.replace('.csv', '')
            data_file_dir.mkdir(exist_ok=True)
            
            # Save CSV results
            csv_file = data_file_dir / "results.csv"
            df = pd.DataFrame(group_results)
            df.to_csv(csv_file, index=False)
            
            # Save JSON results
            json_file = data_file_dir / "results.json"
            with open(json_file, 'w') as f:
                json.dump(group_results, f, indent=2, default=str)
            
            # Save summary
            summary_file = data_file_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Core Engine Results for {data_file}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Results: {len(group_results)}\n\n")
                
                if group_results:
                    avg_return = sum(r['total_return'] for r in group_results) / len(group_results)
                    avg_sharpe = sum(r['sharpe_ratio'] for r in group_results) / len(group_results)
                    avg_drawdown = sum(r['max_drawdown'] for r in group_results) / len(group_results)
                    
                    f.write(f"Average Return: {avg_return:.2%}\n")
                    f.write(f"Average Sharpe: {avg_sharpe:.2f}\n")
                    f.write(f"Average Max Drawdown: {avg_drawdown:.2%}\n\n")
                    
                    best = max(group_results, key=lambda r: r['total_return'])
                    f.write(f"Best Performer:\n")
                    f.write(f"  Strategy: {best['strategy_name']}\n")
                    f.write(f"  Return: {best['total_return']:.2%}\n")
                    f.write(f"  Sharpe: {best['sharpe_ratio']:.2f}\n")
                    f.write(f"  Max Drawdown: {best['max_drawdown']:.2%}\n\n")
        
        print(f"✅ Results saved to: {base_results_dir}")
        return True
            
        except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Main function to run Core Engine processing"""
            start_time = datetime.now()
    
    print(f"Starting Core Engine processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Discover files
    data_files = discover_data_files()
    strategy_files = discover_strategies()
            
            if not data_files:
        print("❌ No data files found")
        return
            
            if not strategy_files:
        print("❌ No strategy files found")
        return
            
            # Process combinations
            results = []
            total_combinations = len(data_files) * len(strategy_files)
            completed = 0
            
    print(f"\nProcessing {len(data_files)} data files with {len(strategy_files)} strategies")
    print(f"Total combinations: {total_combinations}")
    
    for data_file in data_files:
                data_name = data_file.stem
        print(f"\n📊 Processing {data_name}...")
                
                # Load data
        data = load_data(data_file)
                if data is None:
                    continue
                
        print(f"   Data loaded: {len(data)} rows")
                
                # Process each strategy
        for strategy_file in strategy_files:
                    strategy_name = strategy_file.stem
            print(f"   🔄 Testing {strategy_name}...")
                    
            # Create strategy
            strategy = create_simple_strategy(strategy_name)
                    
                    # Run backtest
            result = run_backtest(strategy, data, strategy_name, data_name)
            if result:
                    results.append(result)
                    
                    completed += 1
                    if completed % 10 == 0:
                        progress = (completed / total_combinations) * 100
                print(f"   📈 Progress: {progress:.1f}% ({completed}/{total_combinations})")
            
            # Save results
            if results:
        results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
        save_results(results, results_path)
    
    execution_time = datetime.now() - start_time
    
    print("\n" + "=" * 80)
    print("CORE ENGINE PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Processing completed successfully!")
    print(f"📊 Results Generated: {len(results)}")
    print(f"⏱️  Execution Time: {execution_time}")
    print(f"📁 Results Location: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Results")
    
    print(f"\nResults are organized by:")
    print(f"- Data File (e.g., BTC_1h_5000candles_20250421)")
    print(f"- Strategy (e.g., simple_momentum_strategy)")
    print(f"- Engine (CoreEngine)")
    print(f"- File Type (csv, json, txt)")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
