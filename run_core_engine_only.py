#!/usr/bin/env python3
"""
Run Core Engine Only - Process real data files with strategies
This script runs only the Core Engine against your actual data files and strategies,
saving results organized by data file, strategy, and engine.
"""

import asyncio
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
print("RUNNING CORE ENGINE - REAL DATA ONLY")
print("=" * 80)
print("Processing actual data files with strategies")
print("Results organized by: data_file/strategy/engine")
print("=" * 80)

async def run_core_engine_only():
    """Run Core Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING CORE ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Configure for real data processing
        config = EngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            save_json=True,
            save_csv=True,
            save_plots=False,
            max_data_points=10000,  # Process reasonable amount of data
            chunk_size=1000,
            max_workers=2  # Limit workers for stability
        )
        
        print("Initializing Core Engine...")
        engine = CoreEngine(config)
        
        print("Starting data processing...")
        results = await engine.run()
        
        print(f"✅ Core Engine completed - {len(results)} results generated")
        
        # Show results summary
        if results:
            print(f"\nResults Summary:")
            print(f"- Total results: {len(results)}")
            print(f"- Results saved to: {config.results_path}")
            
            # Show sample results
            if len(results) > 0:
                sample_result = results[0]
                print(f"\nSample Result:")
                print(f"- Strategy: {sample_result.strategy_name}")
                print(f"- Data File: {sample_result.data_file}")
                print(f"- Total Return: {sample_result.total_return:.4f}")
                print(f"- Sharpe Ratio: {sample_result.sharpe_ratio:.4f}")
                print(f"- Max Drawdown: {sample_result.max_drawdown:.4f}")
        
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Core Engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

async def main():
    """Main function to run Core Engine only"""
    start_time = datetime.now()
    
    print(f"Starting Core Engine processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will process your real data files with all available strategies")
    
    try:
        success, result_count = await run_core_engine_only()
        
        execution_time = datetime.now() - start_time
        
        print("\n" + "=" * 80)
        print("CORE ENGINE PROCESSING COMPLETE")
        print("=" * 80)
        
        if success:
            print(f"✅ Core Engine completed successfully!")
            print(f"📊 Results Generated: {result_count}")
            print(f"⏱️  Execution Time: {execution_time}")
            print(f"📁 Results Location: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Results")
            
            print(f"\nResults are organized by:")
            print(f"- Data File (e.g., BTC_1h_5000candles_20250421.csv)")
            print(f"- Strategy (e.g., simple_momentum_strategy)")
            print(f"- Engine (CoreEngine)")
            print(f"- File Type (csv, json)")
            
            print(f"\n🎉 Processing completed successfully!")
        else:
            print(f"❌ Core Engine failed")
            print(f"⏱️  Execution Time: {execution_time}")
            print(f"Check the error messages above for details")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        print(f"⏱️  Execution Time: {datetime.now() - start_time}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
