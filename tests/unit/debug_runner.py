#!/usr/bin/env python3
"""
Debug script for engine runner
"""

import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=== Debug Engine Runner ===")
    
    try:
        print("Step 1: Testing basic imports...")
        import sys
        print(f"Python version: {sys.version}")
        
        print("Step 2: Testing engine registry import...")
        from core.engine_registry import discover_engines
        print("Engine registry import successful")
        
        print("Step 3: Discovering engines...")
        engines = discover_engines()
        print(f"Discovered {len(engines)} engines: {list(engines.keys())}")
        
        print("Step 4: Testing config imports...")
        from configs.basic_config import BasicEngineConfig
        print("BasicEngineConfig import successful")
        
        print("Step 5: Creating config...")
        config = BasicEngineConfig()
        print("Config created successfully")
        
        print("=== Debug Completed Successfully ===")
        
    except Exception as e:
        print(f"Error: {e}")
        print("=== Traceback ===")
        traceback.print_exc()
        print("=== End Traceback ===")

if __name__ == "__main__":
    main()
