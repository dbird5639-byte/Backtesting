#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

print("=== FORCING OUTPUT TEST ===")

from Engines.core_engine import CoreEngine, EngineConfig

print("1. Creating config...")
config = EngineConfig()
config.parallel_workers = 1
config.verbose = True
config.results_path = "Results"

print("2. Creating engine...")
engine = CoreEngine(config)

print("3. Discovering files...")
data_files = engine.discover_data_files()
strategy_files = engine.discover_strategy_files()

print(f"Found {len(data_files)} data files")
print(f"Found {len(strategy_files)} strategy files")

if not data_files or not strategy_files:
    print("ERROR: No files found!")
    sys.exit(1)

print("4. Testing single combination...")
data_file = data_files[0]
strategy_file = strategy_files[0]

print(f"Testing: {data_file.name} + {strategy_file.name}")

# Load data
data = engine.load_data(data_file)
print(f"Data loaded: {data is not None}")

if data is not None:
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")

# Load strategy
strategy_cls = engine.load_strategy(strategy_file)
print(f"Strategy loaded: {strategy_cls is not None}")

if strategy_cls is not None:
    print(f"Strategy class: {strategy_cls.__name__}")

# Run backtest
if data is not None and strategy_cls is not None:
    print("Running backtest...")
    result = engine.run_single_backtest(strategy_cls, data)
    print(f"Backtest result: {result is not None}")
    if result:
        print(f"Return: {result.get('total_return', 'N/A')}")
        print(f"Sharpe: {result.get('sharpe_ratio', 'N/A')}")

print("=== TEST COMPLETED ===")
