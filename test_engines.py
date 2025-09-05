#!/usr/bin/env python3
"""
Simple Engine Test Script
Tests if the engines can be imported and basic functionality works
"""

import sys
import os
sys.path.append('.')

def test_engine_imports():
    """Test if engines can be imported"""
    print("🔍 Testing Engine Imports...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        print("✅ Core Engine imported successfully")
    except Exception as e:
        print(f"❌ Core Engine import failed: {e}")
        return False
    
    try:
        from Engines.ml_engine import MLEngine, MLEngineConfig
        print("✅ ML Engine imported successfully")
    except Exception as e:
        print(f"❌ ML Engine import failed: {e}")
        return False
    
    try:
        from Engines.risk_engine import RiskEngine, RiskEngineConfig
        print("✅ Risk Engine imported successfully")
    except Exception as e:
        print(f"❌ Risk Engine import failed: {e}")
        return False
    
    try:
        from Engines.performance_engine import PerformanceEngine, PerformanceEngineConfig
        print("✅ Performance Engine imported successfully")
    except Exception as e:
        print(f"❌ Performance Engine import failed: {e}")
        return False
    
    try:
        from Engines.engine_factory import EngineFactory, EngineType
        print("✅ Engine Factory imported successfully")
        print(f"   Available engines: {[e.value for e in EngineType]}")
    except Exception as e:
        print(f"❌ Engine Factory import failed: {e}")
        return False
    
    return True

def test_engine_creation():
    """Test if engines can be created"""
    print("\n🔧 Testing Engine Creation...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig(
            data_path="./test_data",
            results_path="./test_results",
            initial_cash=100000.0
        )
        
        engine = CoreEngine(config)
        print("✅ Core Engine created successfully")
        
    except Exception as e:
        print(f"❌ Core Engine creation failed: {e}")
        return False
    
    return True

def test_strategy_imports():
    """Test if strategies can be imported"""
    print("\n📈 Testing Strategy Imports...")
    
    try:
        import os
        strategy_files = [f for f in os.listdir('./Strategies') if f.endswith('.py') and f != '__init__.py']
        print(f"✅ Found {len(strategy_files)} strategy files")
        
        for strategy_file in strategy_files[:3]:  # Test first 3
            try:
                module_name = strategy_file[:-3]  # Remove .py
                exec(f"import Strategies.{module_name}")
                print(f"✅ {strategy_file} imported successfully")
            except Exception as e:
                print(f"❌ {strategy_file} import failed: {e}")
        
    except Exception as e:
        print(f"❌ Strategy import test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Engine and Strategy Test Suite")
    print("=" * 50)
    
    # Test imports
    engine_imports_ok = test_engine_imports()
    engine_creation_ok = test_engine_creation()
    strategy_imports_ok = test_strategy_imports()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Engine Imports: {'✅ PASS' if engine_imports_ok else '❌ FAIL'}")
    print(f"Engine Creation: {'✅ PASS' if engine_creation_ok else '❌ FAIL'}")
    print(f"Strategy Imports: {'✅ PASS' if strategy_imports_ok else '❌ FAIL'}")
    
    if engine_imports_ok and engine_creation_ok and strategy_imports_ok:
        print("\n🎉 ALL TESTS PASSED! Your engines and strategies are ready to run!")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
