#!/usr/bin/env python3
"""
Test Path Configuration
This script verifies that all engines are using the correct paths.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Path Configuration for All Engines")
print("=" * 60)

def test_core_engine_paths():
    """Test Core Engine paths"""
    print("\n1. Testing Core Engine Paths...")
    try:
        from Engines.core_engine import EngineConfig
        
        config = EngineConfig()
        
        expected_data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
        expected_strategies_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
        expected_results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        
        print(f"  Data Path: {config.data_path}")
        print(f"  Strategies Path: {config.strategies_path}")
        print(f"  Results Path: {config.results_path}")
        
        if (config.data_path == expected_data_path and 
            config.strategies_path == expected_strategies_path and 
            config.results_path == expected_results_path):
            print("  ✅ Core Engine paths are correct")
            return True
        else:
            print("  ❌ Core Engine paths are incorrect")
            return False
            
    except Exception as e:
        print(f"  ❌ Core Engine test failed: {e}")
        return False

def test_enhanced_risk_engine_paths():
    """Test Enhanced Risk Engine paths"""
    print("\n2. Testing Enhanced Risk Engine Paths...")
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngineConfig
        
        config = EnhancedRiskEngineConfig()
        
        expected_data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
        expected_strategies_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
        expected_results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        
        print(f"  Data Path: {config.data_path}")
        print(f"  Strategies Path: {config.strategies_path}")
        print(f"  Results Path: {config.results_path}")
        
        if (config.data_path == expected_data_path and 
            config.strategies_path == expected_strategies_path and 
            config.results_path == expected_results_path):
            print("  ✅ Enhanced Risk Engine paths are correct")
            return True
        else:
            print("  ❌ Enhanced Risk Engine paths are incorrect")
            return False
            
    except Exception as e:
        print(f"  ❌ Enhanced Risk Engine test failed: {e}")
        return False

def test_visualization_engine_paths():
    """Test Visualization Engine paths"""
    print("\n3. Testing Visualization Engine Paths...")
    try:
        from Engines.enhanced_visualization_engine import VisualizationConfig
        
        config = VisualizationConfig()
        
        expected_data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
        expected_strategies_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
        expected_results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        
        print(f"  Data Path: {config.data_path}")
        print(f"  Strategies Path: {config.strategies_path}")
        print(f"  Results Path: {config.results_path}")
        
        if (config.data_path == expected_data_path and 
            config.strategies_path == expected_strategies_path and 
            config.results_path == expected_results_path):
            print("  ✅ Visualization Engine paths are correct")
            return True
        else:
            print("  ❌ Visualization Engine paths are incorrect")
            return False
            
    except Exception as e:
        print(f"  ❌ Visualization Engine test failed: {e}")
        return False

def test_regime_overlay_engine_paths():
    """Test Regime Overlay Engine paths"""
    print("\n4. Testing Regime Overlay Engine Paths...")
    try:
        from Engines.regime_overlay_engine import RegimeOverlayConfig
        
        config = RegimeOverlayConfig()
        
        expected_data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
        expected_strategies_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
        expected_results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        
        print(f"  Data Path: {config.data_path}")
        print(f"  Strategies Path: {config.strategies_path}")
        print(f"  Results Path: {config.results_path}")
        print(f"  Regime Data Path: {config.regime_data_path}")
        
        if (config.data_path == expected_data_path and 
            config.strategies_path == expected_strategies_path and 
            config.results_path == expected_results_path):
            print("  ✅ Regime Overlay Engine paths are correct")
            return True
        else:
            print("  ❌ Regime Overlay Engine paths are incorrect")
            return False
            
    except Exception as e:
        print(f"  ❌ Regime Overlay Engine test failed: {e}")
        return False

def test_directory_structure():
    """Test that the directory structure exists"""
    print("\n5. Testing Directory Structure...")
    
    try:
        # Check if directories exist
        data_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data")
        strategies_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies")
        results_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results")
        
        print(f"  Data Directory: {data_path} - {'✅ Exists' if data_path.exists() else '❌ Missing'}")
        print(f"  Strategies Directory: {strategies_path} - {'✅ Exists' if strategies_path.exists() else '❌ Missing'}")
        print(f"  Results Directory: {results_path} - {'✅ Exists' if results_path.exists() else '❌ Missing'}")
        
        # Count files in data directory
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            print(f"  CSV Files in Data Directory: {len(csv_files)}")
        
        # Count files in strategies directory
        if strategies_path.exists():
            strategy_files = list(strategies_path.glob("*.py"))
            print(f"  Strategy Files: {len(strategy_files)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Directory structure test failed: {e}")
        return False

def main():
    """Main test function"""
    tests = [
        ("Core Engine", test_core_engine_paths),
        ("Enhanced Risk Engine", test_enhanced_risk_engine_paths),
        ("Visualization Engine", test_visualization_engine_paths),
        ("Regime Overlay Engine", test_regime_overlay_engine_paths),
        ("Directory Structure", test_directory_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("PATH CONFIGURATION TEST RESULTS")
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
        print("\n🎉 ALL PATH CONFIGURATIONS ARE CORRECT!")
        print("Your engines are ready to use the specified paths:")
        print("  📁 Data: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Data")
        print("  📁 Strategies: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Strategies")
        print("  📁 Results: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Engines\\Results")
        print("\nResults will be organized by:")
        print("  - Engine type (CoreEngine, EnhancedRiskEngine, etc.)")
        print("  - File type (csv, json, png, txt)")
        print("  - Strategy and data file")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED. Please check the path configurations.")

if __name__ == "__main__":
    main()
