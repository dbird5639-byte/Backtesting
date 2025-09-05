#!/usr/bin/env python3
"""
Simple Pipeline Test - Test individual engines
"""

import sys
import os
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def test_core_engine():
    """Test core engine"""
    print("🧪 Testing CoreEngine...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Create config
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        
        # Create engine
        engine = CoreEngine(config)
        
        # Test data discovery
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        print(f"   📊 Found {len(data_files)} data files")
        print(f"   🎯 Found {len(strategy_files)} strategy files")
        
        if len(data_files) > 0 and len(strategy_files) > 0:
            print("✅ CoreEngine test passed")
            return True
        else:
            print("❌ No data or strategy files found")
            return False
            
    except Exception as e:
        print(f"❌ CoreEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting Simple Pipeline Test")
    print("=" * 50)
    
    # Test core engine
    if test_core_engine():
        print("\n✅ CoreEngine test passed! Ready to run pipeline.")
        return True
    else:
        print("\n❌ CoreEngine test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
