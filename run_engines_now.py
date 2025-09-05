#!/usr/bin/env python3
"""
Run Engines Now - Show Results
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=== RUNNING ENGINES NOW ===")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Create config
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        config.results_path = "Results"
        config.max_data_points = 1000  # Limit for faster testing
        
        # Create engine
        engine = CoreEngine(config)
        
        # Discover files
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        print(f"Found {len(data_files)} data files")
        print(f"Found {len(strategy_files)} strategy files")
        
        # Test with first few combinations
        print("\nTesting first 5 combinations...")
        results = []
        
        for i in range(min(5, len(data_files))):
            for j in range(min(5, len(strategy_files))):
                data_file = data_files[i]
                strategy_file = strategy_files[j]
                
                print(f"Testing: {data_file.name} + {strategy_file.name}")
                
                # Load data
                data = engine.load_data(data_file)
                if data is None:
                    print(f"  ❌ Data loading failed")
                    continue
                
                # Load strategy
                strategy_cls = engine.load_strategy(strategy_file)
                if strategy_cls is None:
                    print(f"  ❌ Strategy loading failed")
                    continue
                
                # Run backtest
                try:
                    result = engine.run_single_backtest(strategy_cls, data)
                    if result:
                        print(f"  ✅ Success: Return={result.get('total_return', 0):.2%}, Sharpe={result.get('sharpe_ratio', 0):.2f}")
                        results.append(result)
                    else:
                        print(f"  ❌ Backtest failed")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        # Save results
        if results:
            print(f"\nSaving {len(results)} results...")
            engine.save_results(results)
            print("✅ Results saved!")
        else:
            print("❌ No results to save")
        
        print("\n=== ENGINES COMPLETED ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
