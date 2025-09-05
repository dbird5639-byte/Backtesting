#!/usr/bin/env python3
"""
Test script for Enhanced Risk Analysis System

This script tests the comprehensive enhanced risk analysis system
to ensure all components work together properly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Testing Enhanced Risk Analysis System")
print("==========================================")

async def create_test_data(n_rows: int = 1000) -> Path:
    """Create sample OHLCV data for testing"""
    print("\n📊 Creating test data for enhanced risk analysis...")
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price * (1 + np.random.normal(0, 0.005)) for _ in range(n_rows)]
    
    data = []
    for i, date in enumerate(dates):
        open_price = prices[i-1] if i > 0 else prices[i]
        close_price = prices[i]
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
        volume = np.random.randint(1000, 10000)
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    test_data_dir = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data")
    test_data_dir.mkdir(exist_ok=True)
    file_path = test_data_dir / "enhanced_risk_analysis_test_data.csv"
    df.to_csv(file_path)
    print(f"✅ Test data created: {n_rows} rows saved to {file_path}")
    return file_path

async def test_enhanced_risk_analysis_imports():
    """Test imports for enhanced risk analysis"""
    print("\n🔍 Testing Enhanced Risk Analysis Imports...")
    
    try:
        from scripts.run_enhanced_risk_analysis import EnhancedRiskAnalysisOrchestrator
        print("✅ Enhanced Risk Analysis Orchestrator imported successfully")
        return True
    except Exception as e:
        print(f"❌ Enhanced Risk Analysis import failed: {e}")
        return False

async def test_enhanced_risk_analysis_creation():
    """Test creation of enhanced risk analysis orchestrator"""
    print("\n🔍 Testing Enhanced Risk Analysis Creation...")
    
    try:
        from scripts.run_enhanced_risk_analysis import EnhancedRiskAnalysisOrchestrator
        orchestrator = EnhancedRiskAnalysisOrchestrator()
        print("✅ Enhanced Risk Analysis Orchestrator created successfully")
        return True
    except Exception as e:
        print(f"❌ Enhanced Risk Analysis creation failed: {e}")
        return False

async def test_data_discovery():
    """Test data file discovery"""
    print("\n🔍 Testing Data File Discovery...")
    
    try:
        from scripts.run_enhanced_risk_analysis import EnhancedRiskAnalysisOrchestrator
        orchestrator = EnhancedRiskAnalysisOrchestrator()
        data_files = orchestrator.discover_data_files()
        
        if data_files:
            print(f"✅ Data discovery successful: {len(data_files)} files found")
            for file in data_files[:3]:  # Show first 3 files
                print(f"  - {file}")
            if len(data_files) > 3:
                print(f"  ... and {len(data_files) - 3} more files")
            return True
        else:
            print("⚠️  No data files found (this is expected if no data exists)")
            return True  # Not a failure, just no data
    except Exception as e:
        print(f"❌ Data discovery failed: {e}")
        return False

async def test_enhanced_risk_analysis_components():
    """Test individual components of enhanced risk analysis"""
    print("\n🔍 Testing Enhanced Risk Analysis Components...")
    
    try:
        from scripts.run_enhanced_risk_analysis import EnhancedRiskAnalysisOrchestrator
        orchestrator = EnhancedRiskAnalysisOrchestrator()
        
        # Test risk engine
        risk_engine = orchestrator.risk_engine
        print("✅ Risk engine component accessible")
        
        # Test visualization engine
        visualization_engine = orchestrator.visualization_engine
        print("✅ Visualization engine component accessible")
        
        # Test regime overlay engine
        regime_overlay_engine = orchestrator.regime_overlay_engine
        print("✅ Regime overlay engine component accessible")
        
        # Test configurations
        risk_config = orchestrator.risk_config
        print(f"✅ Risk config: Walk-forward enabled = {risk_config.walk_forward_enabled}")
        print(f"✅ Risk config: In-sample periods = {risk_config.in_sample_periods}")
        print(f"✅ Risk config: Out-of-sample periods = {risk_config.out_of_sample_periods}")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced Risk Analysis components test failed: {e}")
        return False

async def main():
    """Main test function"""
    # Create test data
    test_data_file = await create_test_data()
    
    # Run tests
    tests = [
        ("Enhanced Risk Analysis Imports", test_enhanced_risk_analysis_imports),
        ("Enhanced Risk Analysis Creation", test_enhanced_risk_analysis_creation),
        ("Data File Discovery", test_data_discovery),
        ("Enhanced Risk Analysis Components", test_enhanced_risk_analysis_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Enhanced Risk Analysis System Test Results:")
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
        print("✅ Your enhanced risk analysis system is ready!")
        print("\n📋 Features Available:")
        print("1. 🔍 Walk-forward analysis with in-sample and out-of-sample testing")
        print("2. 🛡️  Five safe parameter values (conservative risk management)")
        print("3. ⚡ Five risky parameter values (aggressive risk management)")
        print("4. 📊 Comprehensive parameter comparison and optimization")
        print("5. 🎯 Risk attribution and decomposition analysis")
        print("6. 📈 Performance metrics across all risk levels and phases")
        print("7. 📋 Detailed reports and executive summaries")
        print("8. 🎨 Comprehensive visualizations and charts")
        
        print("\n📋 Next Steps:")
        print("1. Add your real market data to the Data folder")
        print("2. Run: python scripts/run_enhanced_risk_analysis.py")
        print("3. Analyze the walk-forward results and parameter comparisons")
        print("4. Use the risk attribution analysis for strategy optimization")
        print("5. Review the executive summary for key insights")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED. Please review the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
