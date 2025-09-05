#!/usr/bin/env python3
"""
Debug Data Loading
"""

import sys
from pathlib import Path
import pandas as pd

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("Debugging data loading...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig()
        engine = CoreEngine(config)
        
        # Test data loading
        data_file = Path("Data/BCH_15m_20250830_185939.csv")
        print(f"Loading data from: {data_file}")
        
        data = engine.load_data(data_file)
        if data is None:
            print("❌ Data loading failed")
            return False
        
        print(f"✓ Data loaded successfully")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Data types: {data.dtypes.to_dict()}")
        print(f"First few rows:")
        print(data.head())
        
        # Test data preparation for backtesting
        data_for_backtest = data.copy()
        data_for_backtest.columns = [col.lower() for col in data_for_backtest.columns]
        print(f"\nAfter column conversion:")
        print(f"Columns: {data_for_backtest.columns.tolist()}")
        
        # Check if backtesting requirements are met
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in data_for_backtest.columns]
        if missing:
            print(f"❌ Missing required columns: {missing}")
        else:
            print("✓ All required columns present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Data debugging completed!")
    else:
        print("\n[FAILED] Data debugging failed!")
    sys.exit(0 if success else 1)
