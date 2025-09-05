#!/usr/bin/env python3
"""
Test script to run the basic engine directly
"""

import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=== Testing Basic Engine ===")
    
    try:
        from configs.basic_config import BasicEngineConfig
        print("BasicEngineConfig import successful")
        
        from engines.basic_engine import BasicEngine
        print("BasicEngine import successful")
        
        # Create config
        config = BasicEngineConfig()
        print("Config created successfully")
        
        # Create engine
        engine = BasicEngine(config)
        print("Engine created successfully")
        
        # Try to run
        print("Attempting to run engine...")
        results = engine.run()
        print(f"Engine run completed with {len(results) if results else 0} results")
        
    except Exception as e:
        print(f"Error: {e}")
        print("=== Traceback ===")
        traceback.print_exc()
        print("=== End Traceback ===")

if __name__ == "__main__":
    main()
