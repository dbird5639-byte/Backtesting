#!/usr/bin/env python3
"""
Test Pipeline - Quick test of the optimized pipeline runner
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def test_imports():
    """Test that all engines can be imported"""
    print("🧪 Testing engine imports...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        print("✅ CoreEngine imported successfully")
    except Exception as e:
        print(f"❌ CoreEngine import failed: {e}")
        return False
    
    try:
        from Engines.risk_engine import RiskEngine, RiskEngineConfig
        print("✅ RiskEngine imported successfully")
    except Exception as e:
        print(f"❌ RiskEngine import failed: {e}")
        return False
    
    try:
        from Engines.statistical_engine import StatisticalEngine, StatisticalEngineConfig
        print("✅ StatisticalEngine imported successfully")
    except Exception as e:
        print(f"❌ StatisticalEngine import failed: {e}")
        return False
    
    try:
        from Engines.validation_engine import ValidationEngine, ValidationEngineConfig
        print("✅ ValidationEngine imported successfully")
    except Exception as e:
        print(f"❌ ValidationEngine import failed: {e}")
        return False
    
    return True

def test_core_engine():
    """Test core engine functionality"""
    print("\n🧪 Testing CoreEngine functionality...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Create engine with test config
        config = EngineConfig()
        config.parallel_workers = 2  # Use fewer workers for testing
        config.verbose = True
        
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
            print("❌ CoreEngine test failed - no data or strategy files found")
            return False
            
    except Exception as e:
        print(f"❌ CoreEngine test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Pipeline Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        return False
    
    # Test core engine
    if not test_core_engine():
        print("\n❌ CoreEngine test failed")
        return False
    
    print("\n✅ All tests passed! Pipeline is ready to run.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
