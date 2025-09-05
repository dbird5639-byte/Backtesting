#!/usr/bin/env python3
"""
Test script for the recursive engine runner - simplified version to test basic functionality.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append('.')

def test_basic_functionality():
    """Test basic functionality without heavy imports."""
    print("Testing recursive runner basic functionality...")
    
    # Test paths
    data_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data")
    strategies_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies")
    results_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results")
    
    print(f"Data path exists: {data_path.exists()}")
    print(f"Strategies path exists: {strategies_path.exists()}")
    print(f"Results path exists: {results_path.exists()}")
    
    # Test data file discovery
    if data_path.exists():
        csv_files = list(data_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        if csv_files:
            print(f"Sample files: {[f.name for f in csv_files[:5]]}")
    
    # Test strategy discovery
    if strategies_path.exists():
        strategy_files = list(strategies_path.glob("*.py"))
        strategy_files = [f for f in strategy_files if f.name != "__init__.py"]
        print(f"Found {len(strategy_files)} strategy files")
        if strategy_files:
            print(f"Sample strategies: {[f.stem for f in strategy_files[:5]]}")
    
    # Test results directory structure
    if results_path.exists():
        print(f"Results directory contents: {list(results_path.iterdir())}")
    
    # Test combination generation
    engines = ['CoreEngine', 'EnhancedRiskEngine', 'EnhancedVisualizationEngine']
    strategies = ['momentum_strategy', 'mean_reversion_strategy', 'dma_histogram_momentum_strategy'] if strategies_path.exists() else []
    data_files = [f.name for f in csv_files[:3]] if data_path.exists() and csv_files else []
    
    total_combinations = len(engines) * len(strategies) * len(data_files)
    print(f"Total combinations to test: {total_combinations}")
    
    if total_combinations > 0:
        print("Sample combinations:")
        for i, engine in enumerate(engines[:2]):
            for j, strategy in enumerate(strategies[:2]):
                for k, data_file in enumerate(data_files[:2]):
                    print(f"  {engine} - {strategy} - {data_file}")
                    if i + j + k >= 5:  # Limit output
                        break
                if i + j >= 5:
                    break
            if i >= 5:
                break
    
    print("Basic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()
