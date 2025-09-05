#!/usr/bin/env python3
"""
Simple Core Engine Test
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("Starting Core Engine Test...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        print("Creating config...")
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        
        print("Creating engine...")
        engine = CoreEngine(config)
        
        print("Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        print(f"Found {len(data_files)} data files")
        print(f"Found {len(strategy_files)} strategy files")
        
        if len(data_files) > 0 and len(strategy_files) > 0:
            print("Testing first combination...")
            result = engine.process_file_strategy_combination(data_files[0], strategy_files[0])
            print(f"Got {len(result)} results")
            
            if result:
                print("First result keys:", list(result[0].keys()))
        
        print("Core Engine Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
