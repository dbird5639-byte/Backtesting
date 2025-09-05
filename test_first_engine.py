#!/usr/bin/env python3
"""
Test First Engine Only
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=" * 80)
    print("STARTING CORE ENGINE TEST")
    print("=" * 80)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        print("Creating config...")
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        
        print("Creating engine...")
        engine = CoreEngine(config)
        
        print("Running engine...")
        engine.run()
        
        print("SUCCESS: Core Engine completed!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
    print("CORE ENGINE TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
