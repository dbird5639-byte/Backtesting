#!/usr/bin/env python3
"""
Test Core Engine Only
This script runs just the core engine to see what's happening.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def test_core_engine():
    """Test the core engine with verbose output"""
    print("Testing Core Engine Only")
    print("=" * 50)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Create config with correct paths and verbose logging
        config = EngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            save_json=False,
            save_csv=True,
            save_plots=False,
            log_level="INFO",
            log_to_file=True,
            enable_progress_bar=True,
            max_workers=2,  # Reduce workers for easier debugging
            max_data_points=10000,  # Limit data points for faster testing
            min_data_points=50  # Lower minimum for testing
        )
        
        print(f"Config:")
        print(f"  Data Path: {config.data_path}")
        print(f"  Strategies Path: {config.strategies_path}")
        print(f"  Results Path: {config.results_path}")
        print(f"  Max Workers: {config.max_workers}")
        print(f"  Max Data Points: {config.max_data_points}")
        print(f"  Min Data Points: {config.min_data_points}")
        print(f"  Save CSV: {config.save_csv}")
        print(f"  Save JSON: {config.save_json}")
        print(f"  Save Plots: {config.save_plots}")
        
        # Create and run engine
        engine = CoreEngine(config)
        print("\nStarting Core Engine...")
        results = await engine.run()
        
        print(f"\nCore Engine Results:")
        print(f"Total Results: {len(results)}")
        
        if results:
            print(f"Average Return: {sum(r.total_return for r in results) / len(results):.2%}")
            print(f"Average Sharpe: {sum(r.sharpe_ratio for r in results) / len(results):.2f}")
            
            # Show first few results
            print(f"\nFirst 5 Results:")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. {result.strategy_name} on {result.data_file}")
                print(f"     Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
            
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more results")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_core_engine())
