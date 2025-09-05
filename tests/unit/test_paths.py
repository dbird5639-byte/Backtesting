#!/usr/bin/env python3
"""
Test path resolution
"""

import os
from pathlib import Path

def test_paths():
    """Test path resolution"""
    print("Current working directory:", os.getcwd())
    
    # Test relative paths
    data_path = Path("../fetched_data")
    strategies_path = Path("../Strategies")
    results_path = Path("../Results")
    
    print(f"Data path: {data_path} (exists: {data_path.exists()})")
    print(f"Strategies path: {strategies_path} (exists: {strategies_path.exists()})")
    print(f"Results path: {results_path} (exists: {results_path.exists()})")
    
    # Test absolute paths
    abs_data_path = Path("../fetched_data").resolve()
    abs_strategies_path = Path("../Strategies").resolve()
    abs_results_path = Path("../Results").resolve()
    
    print(f"Absolute data path: {abs_data_path} (exists: {abs_data_path.exists()})")
    print(f"Absolute strategies path: {abs_strategies_path} (exists: {abs_strategies_path.exists()})")
    print(f"Absolute results path: {abs_results_path} (exists: {abs_results_path.exists()})")
    
    # List contents
    if data_path.exists():
        print(f"Data files: {list(data_path.glob('*'))[:5]}")
    
    if strategies_path.exists():
        print(f"Strategy files: {list(strategies_path.glob('*.py'))[:5]}")

if __name__ == "__main__":
    test_paths()
