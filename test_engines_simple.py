#!/usr/bin/env python3
"""
Simple test script to verify engines are working and saving results
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Enhanced Risk Engine - Simple Version")
print("=" * 50)

async def create_simple_test_data():
    """Create simple test data"""
    print("Creating test data...")
    
    # Create simple OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # Generate price data
    base_price = 100.0
    prices = []
    for i in range(1000):
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # Save to test file
    test_file = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data\simple_test_data.csv")
    df.to_csv(test_file)
    print(f"Test data saved to: {test_file}")
    return str(test_file)

async def test_enhanced_risk_engine():
    """Test the enhanced risk engine"""
    print("\nTesting Enhanced Risk Engine...")
    
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig
        
        # Create configuration
        config = EnhancedRiskEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            walk_forward_enabled=True,
            in_sample_periods=100,  # Smaller for testing
            out_of_sample_periods=20,  # Smaller for testing
            min_periods_for_analysis=50,
            save_walk_forward_results=True,
            save_parameter_comparison=True,
            save_risk_attribution=True,
            create_visualizations=True
        )
        
        # Create engine
        engine = EnhancedRiskEngine(config)
        print("Enhanced Risk Engine created successfully")
        
        # Create test data
        test_file = await create_simple_test_data()
        
        # Run analysis on single file
        print("Running walk-forward analysis...")
        print(f"Test file: {test_file}")
        
        # Check if file exists and has data
        import pandas as pd
        test_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"Test data shape: {test_data.shape}")
        print(f"Test data columns: {test_data.columns.tolist()}")
        print(f"Test data index type: {type(test_data.index[0])}")
        
        results = await engine.run_walk_forward_analysis([test_file])
        
        if results and results.get('walk_forward_results'):
            print(f"SUCCESS: Generated {len(results['walk_forward_results'])} results")
            
            # Check if results were saved
            results_dir = Path(config.results_path)
            if results_dir.exists():
                files = list(results_dir.rglob("*"))
                print(f"Results saved to: {results_dir}")
                print(f"Files created: {len(files)}")
                for file in files[:5]:  # Show first 5 files
                    print(f"  - {file.name}")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more files")
            else:
                print("WARNING: Results directory not found")
            
            return True
        else:
            print("ERROR: No results generated")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_enhanced_risk_engine()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Enhanced Risk Engine is working and saving results!")
        print("Check the Results folder for output files.")
    else:
        print("FAILED: Enhanced Risk Engine test failed.")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
