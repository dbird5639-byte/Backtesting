#!/usr/bin/env python3
"""
Test Script for Regime Analysis System

This script tests the new regime analysis and visualization capabilities
to ensure everything is working correctly.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append('.')

def test_regime_analysis_imports():
    """Test if regime analysis components can be imported"""
    print("🔍 Testing Regime Analysis Imports...")
    
    try:
        from scripts.regime_analysis import RegimeAnalyzer, RegimeConfig
        print("✅ Regime Analysis imported successfully")
    except Exception as e:
        print(f"❌ Regime Analysis import failed: {e}")
        return False
    
    try:
        from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
        print("✅ Enhanced Visualization Engine imported successfully")
    except Exception as e:
        print(f"❌ Enhanced Visualization Engine import failed: {e}")
        return False
    
    try:
        from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
        print("✅ Regime Overlay Engine imported successfully")
    except Exception as e:
        print(f"❌ Regime Overlay Engine import failed: {e}")
        return False
    
    return True

def create_test_data():
    """Create test data for regime analysis"""
    print("📊 Creating test data...")
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_days = 200
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data with different regimes
    base_price = 100.0
    prices = [base_price]
    
    # Create different market regimes
    for i in range(n_days):
        if i < 50:  # Bull market
            daily_return = np.random.normal(0.002, 0.015)
        elif i < 100:  # Bear market
            daily_return = np.random.normal(-0.001, 0.020)
        elif i < 150:  # Sideways market
            daily_return = np.random.normal(0.000, 0.010)
        else:  # High volatility
            daily_return = np.random.normal(0.000, 0.030)
        
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices[1:])):
        daily_vol = abs(np.random.normal(0, 0.01))
        high = price * (1 + abs(np.random.normal(0, daily_vol)))
        low = price * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Save test data
    os.makedirs('./test_data', exist_ok=True)
    df.to_csv('./test_data/test_regime_data.csv', index=False)
    print(f"✅ Test data created: {len(df)} rows saved to ./test_data/test_regime_data.csv")
    
    return df

def test_regime_analysis():
    """Test regime analysis functionality"""
    print("\n🔍 Testing Regime Analysis...")
    
    try:
        from scripts.regime_analysis import RegimeAnalyzer, RegimeConfig
        
        # Create configuration
        config = RegimeConfig(
            data_path="./test_data",
            results_path="./test_results/regime_analysis",
            save_csv=True,
            save_json=True,
            save_plots=False,  # Disable plots for testing
            save_heatmaps=False
        )
        
        # Create analyzer
        analyzer = RegimeAnalyzer(config)
        
        # Run analysis
        results = analyzer.run_analysis()
        
        if results:
            print(f"✅ Regime analysis successful: {len(results)} results generated")
            return True
        else:
            print("❌ Regime analysis failed: No results generated")
            return False
            
    except Exception as e:
        print(f"❌ Regime analysis test failed: {e}")
        return False

def test_enhanced_visualization():
    """Test enhanced visualization functionality"""
    print("\n📊 Testing Enhanced Visualization...")
    
    try:
        from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
        
        # Create configuration
        config = VisualizationConfig(
            data_path="./test_data",
            results_path="./test_results/visualizations",
            save_csv=True,
            save_png=False,  # Disable PNG for testing
            save_html=True,
            enable_regime_overlay=True
        )
        
        # Create engine
        engine = EnhancedVisualizationEngine(config)
        print("✅ Enhanced Visualization Engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Visualization test failed: {e}")
        return False

def test_regime_overlay():
    """Test regime overlay functionality"""
    print("\n🎯 Testing Regime Overlay...")
    
    try:
        from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
        
        # Create configuration
        config = RegimeOverlayConfig(
            data_path="./test_data",
            results_path="./test_results/regime_overlay",
            regime_data_path="./test_results/regime_analysis",
            enable_regime_filtering=True,
            enable_strategy_recommendations=True,
            enable_regime_alerts=True
        )
        
        # Create engine
        engine = RegimeOverlayEngine(config)
        print("✅ Regime Overlay Engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Regime Overlay test failed: {e}")
        return False

def test_comprehensive_script():
    """Test comprehensive analysis script"""
    print("\n🚀 Testing Comprehensive Analysis Script...")
    
    try:
        # Check if the script exists and can be imported
        script_path = "./scripts/run_comprehensive_regime_analysis.py"
        if os.path.exists(script_path):
            print("✅ Comprehensive analysis script exists")
            return True
        else:
            print("❌ Comprehensive analysis script not found")
            return False
            
    except Exception as e:
        print(f"❌ Comprehensive script test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Regime Analysis System")
    print("=" * 50)
    
    # Create test data
    create_test_data()
    
    # Test imports
    imports_ok = test_regime_analysis_imports()
    
    # Test regime analysis
    regime_ok = test_regime_analysis()
    
    # Test enhanced visualization
    viz_ok = test_enhanced_visualization()
    
    # Test regime overlay
    overlay_ok = test_regime_overlay()
    
    # Test comprehensive script
    script_ok = test_comprehensive_script()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Regime Analysis: {'✅ PASS' if regime_ok else '❌ FAIL'}")
    print(f"Enhanced Visualization: {'✅ PASS' if viz_ok else '❌ FAIL'}")
    print(f"Regime Overlay: {'✅ PASS' if overlay_ok else '❌ FAIL'}")
    print(f"Comprehensive Script: {'✅ PASS' if script_ok else '❌ FAIL'}")
    
    if all([imports_ok, regime_ok, viz_ok, overlay_ok, script_ok]):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your regime analysis system is ready!")
        print("\n📋 Next Steps:")
        print("1. Add your real market data to the Data folder")
        print("2. Run: python scripts/run_comprehensive_regime_analysis.py")
        print("3. Check the Results folder for organized outputs")
        print("4. Use the bot integration files for intelligent decision-making")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
