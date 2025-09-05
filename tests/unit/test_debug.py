#!/usr/bin/env python3
"""
Debug script to test basic functionality
"""

import sys
import traceback

def main():
    print("=== Debug Script Starting ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {sys.path[0]}")
    
    try:
        print("Testing imports...")
        from core.engine_registry import discover_engines
        print("✅ Engine registry import successful")
        
        engines = discover_engines()
        print(f"✅ Discovered {len(engines)} engines: {list(engines.keys())}")
        
        print("=== Debug Script Completed Successfully ===")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("=== Traceback ===")
        traceback.print_exc()
        print("=== End Traceback ===")

if __name__ == "__main__":
    main()
