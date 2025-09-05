#!/usr/bin/env python3
"""
Test script to check all key imports
"""

import sys
import traceback

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: {module_name} - {e}")
        return False

def main():
    print("=== Testing Key Imports ===")
    
    # Test basic Python modules
    test_import("pandas", "Pandas")
    test_import("numpy", "NumPy")
    test_import("backtesting", "Backtesting Library")
    test_import("talib", "TA-Lib")
    
    # Test our custom modules
    try:
        sys.path.insert(0, "Engines")
        print("\n=== Testing Custom Module Imports ===")
        
        test_import("core.engine_registry", "Engine Registry")
        test_import("configs.engine_config", "Engine Config")
        test_import("configs.basic_config", "Basic Config")
        test_import("engines.basic_engine", "Basic Engine")
        
    except Exception as e:
        print(f"Error testing custom modules: {e}")
        traceback.print_exc()
    
    print("\n=== Import Test Completed ===")

if __name__ == "__main__":
    main()
