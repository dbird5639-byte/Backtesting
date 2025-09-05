#!/usr/bin/env python3
"""
Run Single Engine Test
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=" * 80)
    print("STARTING SINGLE ENGINE TEST")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        print("Importing CoreEngine...")
        from Engines.core_engine import CoreEngine, EngineConfig
        print("SUCCESS: Import completed")
        
        print("Creating config...")
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        print("SUCCESS: Config created")
        
        print("Creating engine...")
        engine = CoreEngine(config)
        print("SUCCESS: Engine created")
        
        print("Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        print(f"SUCCESS: Found {len(data_files)} data files, {len(strategy_files)} strategy files")
        
        if len(data_files) > 0 and len(strategy_files) > 0:
            print("Processing first combination...")
            result = engine.process_file_strategy_combination(data_files[0], strategy_files[0])
            print(f"SUCCESS: Processed combination, got {len(result)} results")
            
            if result:
                print("First result keys:", list(result[0].keys()))
        else:
            print("ERROR: No files found")
        
        print("SUCCESS: Single engine test completed!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
    print("SINGLE ENGINE TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
