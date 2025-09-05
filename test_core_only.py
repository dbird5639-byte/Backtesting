#!/usr/bin/env python3
"""
Test Core Engine Only
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("🚀 Testing Core Engine Only...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        
        print("✅ Engine imported successfully")
        
        engine = CoreEngine(config)
        print("✅ Engine created successfully")
        
        print("🚀 Starting engine run...")
        engine.run()
        
        print("✅ Core Engine completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
