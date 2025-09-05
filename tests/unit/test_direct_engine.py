#!/usr/bin/env python3
"""
Test script to run the basic engine directly
"""

import sys
import traceback
from pathlib import Path

# Add the Engines directory to the path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=== Testing Basic Engine Directly ===")
    
    try:
        print("Step 1: Testing imports...")
        from configs.basic_config import BasicEngineConfig
        print("BasicEngineConfig import successful")
        
        from engines.basic_engine import BasicEngine
        print("BasicEngine import successful")
        
        print("Step 2: Creating config...")
        config = BasicEngineConfig()
        print("Config created successfully")
        
        print("Step 3: Creating engine...")
        engine = BasicEngine(config)
        print("Engine created successfully")
        
        print("Step 4: Running engine...")
        results = engine.run()
        print(f"Engine run completed with {len(results) if results else 0} results")
        
        print("=== Test Completed Successfully ===")
        
    except Exception as e:
        print(f"Error: {e}")
        print("=== Traceback ===")
        traceback.print_exc()
        print("=== End Traceback ===")

if __name__ == "__main__":
    main()
