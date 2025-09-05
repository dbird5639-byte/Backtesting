#!/usr/bin/env python3
"""
Run Real Data Engines - Process actual data files with strategies
This script runs engines against your real data files and strategies,
saving results organized by data file, strategy, and engine.
"""

import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("RUNNING REAL DATA ENGINES")
print("=" * 80)
print("Processing actual data files with strategies")
print("Results organized by: data_file/strategy/engine")
print("=" * 80)

async def run_core_engine_real_data():
    """Run Core Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING CORE ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Configure for real data processing
        config = EngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            save_json=True,
            save_csv=True,
            save_plots=False,
            max_data_points=50000,  # Process more data
            chunk_size=1000
        )
        
        engine = CoreEngine(config)
        results = await engine.run()
        
        print(f"✅ Core Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Core Engine failed: {e}")
        return False, 0

async def run_enhanced_risk_engine_real_data():
    """Run Enhanced Risk Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING ENHANCED RISK ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig
        
        # Configure for real data processing
        config = EnhancedRiskEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            walk_forward_config={
                'train_size': 0.7,
                'test_size': 0.3,
                'step_size': 30
            }
        )
        
        engine = EnhancedRiskEngine(config)
        results = await engine.run()
        
        print(f"✅ Enhanced Risk Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Enhanced Risk Engine failed: {e}")
        return False, 0

async def run_ml_engine_real_data():
    """Run ML Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING ML ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.ml_engine import MLEngine, MLEngineConfig
        
        # Configure for real data processing
        config = MLEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            test_size=0.2,
            validation_size=0.2
        )
        
        engine = MLEngine(config)
        results = await engine.run()
        
        print(f"✅ ML Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ ML Engine failed: {e}")
        return False, 0

async def run_performance_engine_real_data():
    """Run Performance Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.performance_engine import PerformanceEngine, PerformanceEngineConfig
        
        # Configure for real data processing
        config = PerformanceEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
        )
        
        engine = PerformanceEngine(config)
        results = await engine.run()
        
        print(f"✅ Performance Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Performance Engine failed: {e}")
        return False, 0

async def run_portfolio_engine_real_data():
    """Run Portfolio Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING PORTFOLIO ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.portfolio_engine import PortfolioEngine, PortfolioEngineConfig
        
        # Configure for real data processing
        config = PortfolioEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
        )
        
        engine = PortfolioEngine(config)
        results = await engine.run()
        
        print(f"✅ Portfolio Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Portfolio Engine failed: {e}")
        return False, 0

async def run_validation_engine_real_data():
    """Run Validation Engine against real data files and strategies"""
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION ENGINE - REAL DATA")
    print("=" * 60)
    
    try:
        from Engines.validation_engine import ValidationEngine, ValidationEngineConfig
        
        # Configure for real data processing
        config = ValidationEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
        )
        
        engine = ValidationEngine(config)
        results = await engine.run()
        
        print(f"✅ Validation Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Validation Engine failed: {e}")
        return False, 0

async def main():
    """Main function to run engines against real data"""
    start_time = datetime.now()
    
    # Define engines to run (excluding regime engines as requested)
    engines = [
        ("Core Engine", run_core_engine_real_data),
        ("Enhanced Risk Engine", run_enhanced_risk_engine_real_data),
        ("ML Engine", run_ml_engine_real_data),
        ("Performance Engine", run_performance_engine_real_data),
        ("Portfolio Engine", run_portfolio_engine_real_data),
        ("Validation Engine", run_validation_engine_real_data)
    ]
    
    results = {}
    successful_engines = 0
    total_results = 0
    
    print(f"Starting processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing {len(engines)} engines against real data files and strategies")
    
    for engine_name, engine_func in engines:
        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {engine_name}")
            print(f"{'='*80}")
            
            success, result_count = await engine_func()
            results[engine_name] = {"success": success, "results": result_count}
            
            if success:
                successful_engines += 1
                total_results += result_count
                print(f"✅ {engine_name} completed successfully with {result_count} results")
            else:
                print(f"❌ {engine_name} failed")
                
        except Exception as e:
            print(f"❌ {engine_name} failed with exception: {e}")
            results[engine_name] = {"success": False, "results": 0}
    
    execution_time = datetime.now() - start_time
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - REAL DATA PROCESSING")
    print("=" * 80)
    
    for engine_name, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{engine_name}: {status} ({result['results']} results)")
    
    print(f"\nEngines Completed Successfully: {successful_engines}/{len(engines)}")
    print(f"Total Results Generated: {total_results}")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results Location: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Results")
    
    # Show results organization
    print(f"\nResults are organized by:")
    print(f"- Data File (e.g., BTC_1h_5000candles_20250421.csv)")
    print(f"- Strategy (e.g., simple_momentum_strategy)")
    print(f"- Engine (e.g., CoreEngine, EnhancedRiskEngine)")
    print(f"- File Type (csv, json, txt)")
    
    if successful_engines < len(engines):
        print(f"\n⚠️  {len(engines) - successful_engines} engines failed. Check the logs above for details.")
    else:
        print("\n🎉 All engines completed successfully!")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
