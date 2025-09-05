#!/usr/bin/env python3
"""
Run Core Engine Simple Test
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=" * 80)
    print("RUNNING CORE ENGINE - ALL STRATEGIES vs ALL DATA")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        print("Creating Core Engine...")
        config = EngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        config.results_path = "Results"
        
        engine = CoreEngine(config)
        
        print("Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        print(f"Found {len(data_files)} data files")
        print(f"Found {len(strategy_files)} strategy files")
        print(f"Total combinations: {len(data_files) * len(strategy_files)}")
        
        if not data_files or not strategy_files:
            print("ERROR: No files found")
            return False
        
        print("\nStarting execution...")
        start_time = time.time()
        
        # Run the engine
        engine.run()
        
        execution_time = time.time() - start_time
        
        print(f"\nExecution completed in {execution_time:.2f} seconds")
        
        # Check Results directory
        results_dir = Path("Results")
        if results_dir.exists():
            print("\nResults directory structure:")
            for item in results_dir.iterdir():
                if item.is_dir():
                    print(f"  [DIR] {item.name}/")
                    if "Engine_CoreEngine" in item.name:
                        for subitem in item.iterdir():
                            if subitem.is_dir():
                                print(f"    [DIR] {subitem.name}/")
                                file_count = len(list(subitem.iterdir()))
                                print(f"      [FILES] {file_count} items")
                            else:
                                print(f"    [FILE] {subitem.name}")
                    else:
                        print(f"    [FILE] {item.name}")
        
        print("\n[SUCCESS] Core Engine execution completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n[SUCCESS] Core Engine completed with organized results!")
        else:
            print("\n[FAILED] Core Engine execution failed!")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Execution failed: {e}")
        sys.exit(1)

