#!/usr/bin/env python3
"""
Debug Test - Simple test to see what's happening
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("🚀 Starting debug test...")
    
    try:
        print("1. Importing...")
        from Engines.core_engine import CoreEngine, EngineConfig
        print("✅ Import successful")
        
        print("2. Creating config...")
        config = EngineConfig()
        print("✅ Config created")
        
        print("3. Creating engine...")
        engine = CoreEngine(config)
        print("✅ Engine created")
        
        print("4. Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        print(f"✅ Found {len(data_files)} data files, {len(strategy_files)} strategy files")
        
        if len(data_files) > 0 and len(strategy_files) > 0:
            print("5. Testing single combination...")
            result = engine.process_file_strategy_combination(data_files[0], strategy_files[0])
            print(f"✅ Processed combination, got {len(result)} results")
        else:
            print("❌ No files found")
        
        print("✅ Debug test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
