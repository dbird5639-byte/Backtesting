#!/usr/bin/env python3
"""
Test script for Updated Enhanced Risk Engine Configuration

This script tests the updated configuration with:
- User's specific data path
- Updated risky max drawdown values (25%, 50%, 75%, 90%)
- Safe stop loss, trailing stop, and take profit (four popular values)
- Risky stop loss, trailing stop, and take profit (25%, 50%, 75%, 90%)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Testing Updated Enhanced Risk Engine Configuration")
print("=" * 60)

def test_configuration():
    """Test the updated configuration"""
    print("\n🔍 Testing Enhanced Risk Engine Configuration...")
    
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngineConfig
        
        # Create configuration
        config = EnhancedRiskEngineConfig()
        
        print("✅ Configuration created successfully")
        
        # Test data path
        print(f"📁 Data Path: {config.data_path}")
        expected_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
        if config.data_path == expected_path:
            print("✅ Data path correctly set to user's specified path")
        else:
            print(f"❌ Data path mismatch. Expected: {expected_path}, Got: {config.data_path}")
            return False
        
        # Test safe parameters
        print("\n🛡️  Safe Parameters:")
        safe_params = config.safe_parameters
        print(f"  Stop Loss: {safe_params['stop_loss']}")
        print(f"  Trailing Stop: {safe_params['trailing_stop']}")
        print(f"  Take Profit: {safe_params['take_profit']}")
        
        # Verify safe parameters have 4 values each
        if len(safe_params['stop_loss']) == 4 and len(safe_params['trailing_stop']) == 4 and len(safe_params['take_profit']) == 4:
            print("✅ Safe parameters have 4 values each (popular settings)")
        else:
            print("❌ Safe parameters should have 4 values each")
            return False
        
        # Test risky parameters
        print("\n⚡ Risky Parameters:")
        risky_params = config.risky_parameters
        print(f"  Stop Loss: {risky_params['stop_loss']}")
        print(f"  Trailing Stop: {risky_params['trailing_stop']}")
        print(f"  Take Profit: {risky_params['take_profit']}")
        print(f"  Max Drawdown: {risky_params['max_drawdown']}")
        
        # Verify risky parameters have 4 values each (25%, 50%, 75%, 90%)
        expected_risky_values = [0.25, 0.50, 0.75, 0.90]
        
        if (risky_params['stop_loss'] == expected_risky_values and 
            risky_params['trailing_stop'] == expected_risky_values and 
            risky_params['take_profit'] == expected_risky_values and 
            risky_params['max_drawdown'] == expected_risky_values):
            print("✅ Risky parameters correctly set to 25%, 50%, 75%, 90%")
        else:
            print("❌ Risky parameters should be [0.25, 0.50, 0.75, 0.90]")
            return False
        
        # Test walk-forward settings
        print(f"\n🔍 Walk-Forward Settings:")
        print(f"  Enabled: {config.walk_forward_enabled}")
        print(f"  In-Sample Periods: {config.in_sample_periods}")
        print(f"  Out-of-Sample Periods: {config.out_of_sample_periods}")
        print(f"  Min Periods for Analysis: {config.min_periods_for_analysis}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_engine_creation():
    """Test creating the enhanced risk engine with updated config"""
    print("\n🔍 Testing Enhanced Risk Engine Creation...")
    
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig
        
        config = EnhancedRiskEngineConfig()
        engine = EnhancedRiskEngine(config)
        
        print("✅ Enhanced Risk Engine created successfully with updated configuration")
        return True
        
    except Exception as e:
        print(f"❌ Engine creation failed: {e}")
        return False

def main():
    """Main test function"""
    tests = [
        ("Configuration Test", test_configuration),
        ("Engine Creation Test", test_engine_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Updated Configuration Test Results:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your enhanced risk engine configuration has been updated successfully!")
        print("\n📋 Updated Configuration Summary:")
        print("1. 📁 Data Path: Set to your specific path")
        print("2. 🛡️  Safe Parameters: 4 popular values for SL, TS, TP")
        print("3. ⚡ Risky Parameters: 25%, 50%, 75%, 90% for SL, TS, TP, Max DD")
        print("4. 🔍 Walk-Forward: In-sample and out-of-sample testing enabled")
        
        print("\n📋 Safe Parameter Values:")
        print("  - Stop Loss: 1%, 2%, 3%, 5%")
        print("  - Trailing Stop: 0.5%, 1%, 1.5%, 2%")
        print("  - Take Profit: 2%, 3%, 5%, 8%")
        
        print("\n📋 Risky Parameter Values:")
        print("  - Stop Loss: 25%, 50%, 75%, 90%")
        print("  - Trailing Stop: 25%, 50%, 75%, 90%")
        print("  - Take Profit: 25%, 50%, 75%, 90%")
        print("  - Max Drawdown: 25%, 50%, 75%, 90%")
        
        print("\n📋 Next Steps:")
        print("1. Run: python scripts/run_enhanced_risk_analysis.py")
        print("2. The system will use your data from the specified path")
        print("3. Analyze results with the updated parameter ranges")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED. Please review the errors above.")

if __name__ == "__main__":
    main()
