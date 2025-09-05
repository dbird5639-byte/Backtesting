#!/usr/bin/env python3
"""
Debug Backtesting Library
"""

import sys
from pathlib import Path
import pandas as pd
from backtesting import Backtest, Strategy

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

class TestStrategy(Strategy):
    def init(self):
        pass
    
    def next(self):
        pass

def main():
    print("Debugging backtesting library...")
    
    try:
        # Load data
        data_file = Path("Data/BCH_15m_20250830_185939.csv")
        df = pd.read_csv(data_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Prepare data for backtesting
        data_for_backtest = df.copy()
        data_for_backtest.columns = [col.lower() for col in data_for_backtest.columns]
        
        # Set index to datetime
        data_for_backtest['date'] = pd.to_datetime(data_for_backtest['date'])
        data_for_backtest = data_for_backtest.set_index('date')
        
        print(f"Prepared data shape: {data_for_backtest.shape}")
        print(f"Prepared columns: {data_for_backtest.columns.tolist()}")
        print(f"Index type: {type(data_for_backtest.index)}")
        print(f"First few rows:")
        print(data_for_backtest.head())
        
        # Try to create Backtest
        print("\nCreating Backtest...")
        bt = Backtest(data_for_backtest, TestStrategy, cash=100000, commission=0.001)
        print("✓ Backtest created successfully")
        
        # Try to run it
        print("Running backtest...")
        stats = bt.run()
        print("✓ Backtest ran successfully")
        print(f"Results: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Backtest debugging completed!")
    else:
        print("\n[FAILED] Backtest debugging failed!")
    sys.exit(0 if success else 1)
