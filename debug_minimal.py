#!/usr/bin/env python3
"""
Minimal Debug Test
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("Starting minimal debug test...")
    
    try:
        print("1. Importing CoreEngine...")
        from Engines.core_engine import CoreEngine, EngineConfig
        print("   ✓ Import successful")
        
        print("2. Creating config...")
        config = EngineConfig()
        config.parallel_workers = 1
        config.verbose = True
        config.results_path = "Results"
        print("   ✓ Config created")
        
        print("3. Creating engine...")
        engine = CoreEngine(config)
        print("   ✓ Engine created")
        
        print("4. Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        print(f"   ✓ Found {len(data_files)} data files")
        print(f"   ✓ Found {len(strategy_files)} strategy files")
        
        if not data_files or not strategy_files:
            print("   ❌ No files found!")
            return False
        
        print("5. Testing single combination...")
        # Test just one combination
        data_file = data_files[0]
        strategy_file = strategy_files[0]
        print(f"   Testing: {data_file.name} + {strategy_file.name}")
        
        results = engine.process_file_strategy_combination(data_file, strategy_file)
        print(f"   ✓ Processed, got {len(results)} results")
        
        print("6. Testing save results...")
        if results:
            engine.save_results(results)
            print("   ✓ Results saved")
        
        print("\n[SUCCESS] Minimal test completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Debug test passed!")
    else:
        print("\n[FAILED] Debug test failed!")
    sys.exit(0 if success else 1)
