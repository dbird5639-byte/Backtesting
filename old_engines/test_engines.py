#!/usr/bin/env python3
"""
Test script to verify that all backtesting engines can be imported and instantiated correctly.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import the engines
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all engines can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from base_engine import BaseEngine, EngineConfig
        print("✅ BaseEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BaseEngine: {e}")
        return False
    
    try:
        from basic_engine import BasicEngine, BasicEngineConfig
        print("✅ BasicEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BasicEngine: {e}")
        return False
    
    try:
        from risk_managed_engine import RiskManagedEngine, RiskManagedEngineConfig
        print("✅ RiskManagedEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RiskManagedEngine: {e}")
        return False
    
    try:
        from statistical_engine import StatisticalEngine, StatisticalEngineConfig
        print("✅ StatisticalEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import StatisticalEngine: {e}")
        return False
    
    try:
        from walkforward_engine import WalkforwardEngine, WalkforwardConfig
        print("✅ WalkforwardEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import WalkforwardEngine: {e}")
        return False
    
    try:
        from alpha_engine import AlphaEngine, AlphaEngineConfig
        print("✅ AlphaEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AlphaEngine: {e}")
        return False
    
    try:
        from pipeline_engine import PipelineEngine, PipelineEngineConfig
        print("✅ PipelineEngine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PipelineEngine: {e}")
        return False
    
    return True

def test_instantiation():
    """Test that all engines can be instantiated"""
    print("\n🔍 Testing instantiation...")
    
    try:
        from basic_engine import BasicEngine, BasicEngineConfig
        config = BasicEngineConfig()
        engine = BasicEngine(config)
        print("✅ BasicEngine instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate BasicEngine: {e}")
        return False
    
    try:
        from risk_managed_engine import RiskManagedEngine, RiskManagedEngineConfig
        config = RiskManagedEngineConfig()
        engine = RiskManagedEngine(config)
        print("✅ RiskManagedEngine instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate RiskManagedEngine: {e}")
        return False
    
    try:
        from statistical_engine import StatisticalEngine, StatisticalEngineConfig
        config = StatisticalEngineConfig()
        engine = StatisticalEngine(config)
        print("✅ StatisticalEngine instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate StatisticalEngine: {e}")
        return False
    
    try:
        from walkforward_engine import WalkforwardEngine, WalkforwardConfig
        config = WalkforwardConfig()
        engine = WalkforwardEngine(config)
        print("✅ WalkforwardEngine instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate WalkforwardEngine: {e}")
        return False
    
    try:
        from alpha_engine import AlphaEngine, AlphaEngineConfig
        config = AlphaEngineConfig()
        engine = AlphaEngine(config)
        print("✅ AlphaEngine instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate AlphaEngine: {e}")
        return False
    
    try:
        from pipeline_engine import PipelineEngine, PipelineEngineConfig
        config = PipelineEngineConfig()
        engine = PipelineEngine(config)
        print("✅ PipelineEngine instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate PipelineEngine: {e}")
        return False
    
    return True

def test_configuration():
    """Test that configurations work correctly"""
    print("\n🔍 Testing configurations...")
    
    try:
        from basic_engine import BasicEngineConfig
        config = BasicEngineConfig(
            initial_cash=50000.0,
            commission=0.001,
            backtest_timeout=120
        )
        print("✅ BasicEngineConfig created successfully")
    except Exception as e:
        print(f"❌ Failed to create BasicEngineConfig: {e}")
        return False
    
    try:
        from risk_managed_engine import RiskManagedEngineConfig
        config = RiskManagedEngineConfig(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            enable_walkforward_optimization=True
        )
        print("✅ RiskManagedEngineConfig created successfully")
    except Exception as e:
        print(f"❌ Failed to create RiskManagedEngineConfig: {e}")
        return False
    
    try:
        from walkforward_engine import WalkforwardConfig
        config = WalkforwardConfig(
            train_size=800,
            test_size=150,
            enable_regime_analysis=True
        )
        print("✅ WalkforwardConfig created successfully")
    except Exception as e:
        print(f"❌ Failed to create WalkforwardConfig: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Backtesting Engines Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test instantiation
    if not test_instantiation():
        print("\n❌ Instantiation tests failed!")
        return False
    
    # Test configuration
    if not test_configuration():
        print("\n❌ Configuration tests failed!")
        return False
    
    print("\n✅ All tests passed!")
    print("\n🎉 The backtesting engines are ready to use!")
    print("\nTo run the engines, use:")
    print("  python run_engines.py")
    print("  python example_usage.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 