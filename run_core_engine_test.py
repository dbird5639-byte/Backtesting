#!/usr/bin/env python3
"""
Run Core Engine Test with Organized Results
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=" * 80)
    print("RUNNING CORE ENGINE WITH ORGANIZED RESULTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        print("Creating Core Engine configuration...")
        config = EngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        config.results_path = "Results"
        
        print("Creating Core Engine...")
        engine = CoreEngine(config)
        
        print("Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        print(f"Found {len(data_files)} data files")
        print(f"Found {len(strategy_files)} strategy files")
        print(f"Total combinations: {len(data_files) * len(strategy_files)}")
        
        if not data_files or not strategy_files:
            print("ERROR: No data files or strategy files found")
            return False
        
        print("\nStarting Core Engine execution...")
        start_time = time.time()
        
        # Run the engine
        engine.run()
        
        execution_time = time.time() - start_time
        
        print(f"\nCore Engine completed in {execution_time:.2f} seconds")
        
        # Check Results directory
        results_dir = Path("Results")
        if results_dir.exists():
            print("\nResults directory structure:")
            for item in results_dir.iterdir():
                if item.is_dir() and "Engine_CoreEngine" in item.name:
                    print(f"  📁 {item.name}/")
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            print(f"    📁 {subitem.name}/")
                            for subsubitem in subitem.iterdir():
                                if subsubitem.is_dir():
                                    print(f"      📁 {subsubitem.name}/")
                                    file_count = len(list(subsubitem.iterdir()))
                                    print(f"        📄 {file_count} files")
                                else:
                                    print(f"      📄 {subsubitem.name}")
                        else:
                            print(f"    📄 {subitem.name}")
        
        print("\n✅ Core Engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 SUCCESS: Core Engine with organized results completed!")
        else:
            print("\n💥 FAILED: Core Engine test failed!")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        sys.exit(1)
