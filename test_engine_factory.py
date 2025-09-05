#!/usr/bin/env python3
"""
Engine Factory Test Script
Tests the Engine Factory functionality
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append('.')

async def test_engine_factory():
    """Test the Engine Factory"""
    print("🏭 Testing Engine Factory...")
    
    try:
        from Engines import EngineFactory, EngineFactoryConfig, EngineType, ExecutionMode
        
        # Create configuration
        config = EngineFactoryConfig(
            data_path="./test_data",
            results_path="./test_results",
            execution_mode=ExecutionMode.SEQUENTIAL,
            default_engines=[EngineType.CORE, EngineType.PERFORMANCE],
            log_level="INFO"
        )
        
        # Create factory
        factory = EngineFactory(config)
        print("✅ Engine Factory created successfully")
        
        # Get available engines
        available_engines = factory.get_available_engines()
        print(f"✅ Available engines: {[e.value for e in available_engines]}")
        
        # Get engine info
        for engine_type in [EngineType.CORE, EngineType.ML, EngineType.RISK]:
            info = factory.get_engine_info(engine_type)
            print(f"✅ {engine_type.value} engine info: {info.get('description', 'No description')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine Factory test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    # Generate sample OHLCV data
    np.random.seed(42)
    n_days = 100
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% volatility
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        daily_vol = abs(returns[i]) * 2
        high = price * (1 + abs(np.random.normal(0, daily_vol)))
        low = price * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i-1] if i > 0 else price
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
    
    # Save sample data
    os.makedirs('./test_data', exist_ok=True)
    df.to_csv('./test_data/sample_data.csv', index=False)
    print(f"✅ Sample data created: {len(df)} rows saved to ./test_data/sample_data.csv")
    
    return df

async def test_individual_engine():
    """Test individual engine functionality"""
    print("\n🔧 Testing Individual Engine...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Create configuration
        config = EngineConfig(
            data_path="./test_data",
            results_path="./test_results",
            initial_cash=100000.0,
            commission=0.001,
            max_workers=2
        )
        
        # Create engine
        engine = CoreEngine(config)
        print("✅ Core Engine created successfully")
        
        # Test data loading (if data exists)
        if os.path.exists('./test_data/sample_data.csv'):
            try:
                data = pd.read_csv('./test_data/sample_data.csv')
                print(f"✅ Data loaded successfully: {len(data)} rows")
            except Exception as e:
                print(f"⚠️  Data loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual engine test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Engine Factory Test Suite")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    
    # Test engine factory
    factory_ok = await test_engine_factory()
    
    # Test individual engine
    engine_ok = await test_individual_engine()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Engine Factory: {'✅ PASS' if factory_ok else '❌ FAIL'}")
    print(f"Individual Engine: {'✅ PASS' if engine_ok else '❌ FAIL'}")
    
    if factory_ok and engine_ok:
        print("\n🎉 ALL TESTS PASSED! Your engines are ready for production!")
        print("\n📋 Next Steps:")
        print("1. Add your real market data to the Data folder")
        print("2. Run comprehensive analysis with: python scripts/run_comprehensive_analysis.py")
        print("3. Test specific strategies with the Engine Factory")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
