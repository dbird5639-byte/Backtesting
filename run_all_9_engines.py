#!/usr/bin/env python3
"""
Run All 9 Engines - Comprehensive Test
This script runs all 9 available backtesting engines in the proper sequence:
1. Core Engine (Basic) - CSV results only
2. Enhanced Risk Engine - Walk-forward analysis
3. Enhanced Visualization Engine - Charts and visualizations
4. Regime Detection Engine - Historical regime detection
5. Regime Overlay Engine - Regime intelligence integration
6. ML Engine - Machine learning strategy optimization
7. Performance Engine - Advanced performance analytics
8. Portfolio Engine - Multi-strategy portfolio management
9. Validation Engine - Comprehensive strategy validation
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("RUNNING ALL 9 ENGINES - COMPREHENSIVE TEST")
print("=" * 80)
print("This will run:")
print("1. Core Engine (Basic) - CSV results only")
print("2. Enhanced Risk Engine - Walk-forward analysis")
print("3. Enhanced Visualization Engine - Charts and visualizations")
print("4. Regime Detection Engine - Historical regime detection")
print("5. Regime Overlay Engine - Regime intelligence integration")
print("6. ML Engine - Machine learning strategy optimization")
print("7. Performance Engine - Advanced performance analytics")
print("8. Portfolio Engine - Multi-strategy portfolio management")
print("9. Validation Engine - Comprehensive strategy validation")
print("=" * 80)

async def run_core_engine():
    """Run the basic core engine (CSV only)"""
    print("\n" + "=" * 60)
    print("STEP 1: RUNNING CORE ENGINE (BASIC)")
    print("=" * 60)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            save_json=False,
            save_csv=True,
            save_plots=False
        )
        
        engine = CoreEngine(config)
        results = await engine.run()
        print("✅ Core Engine completed - CSV results saved")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Core Engine failed: {e}")
        return False, 0

async def run_enhanced_risk_engine():
    """Run the enhanced risk engine with walk-forward analysis"""
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING ENHANCED RISK ENGINE")
    print("=" * 60)
    
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig
        
        config = EnhancedRiskEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=5,  # Limit for testing
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

async def run_enhanced_visualization_engine():
    """Run the enhanced visualization engine"""
    print("\n" + "=" * 60)
    print("STEP 3: RUNNING ENHANCED VISUALIZATION ENGINE")
    print("=" * 60)
    
    try:
        from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
        
        config = VisualizationConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3  # Limit for testing
        )
        
        engine = EnhancedVisualizationEngine(config)
        results = await engine.run()
        print("✅ Enhanced Visualization Engine completed - Charts generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Enhanced Visualization Engine failed: {e}")
        return False, 0

async def run_regime_detection_engine():
    """Run the regime detection engine"""
    print("\n" + "=" * 60)
    print("STEP 4: RUNNING REGIME DETECTION ENGINE")
    print("=" * 60)
    
    try:
        from Engines.regime_detection_engine import RegimeDetectionEngine, RegimeConfig
        
        config = RegimeConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3,  # Limit for testing
            n_regimes=3
        )
        
        engine = RegimeDetectionEngine(config)
        results = await engine.run()
        print(f"✅ Regime Detection Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Regime Detection Engine failed: {e}")
        return False, 0

async def run_regime_overlay_engine():
    """Run the regime overlay engine"""
    print("\n" + "=" * 60)
    print("STEP 5: RUNNING REGIME OVERLAY ENGINE")
    print("=" * 60)
    
    try:
        from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
        
        config = RegimeOverlayConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3  # Limit for testing
        )
        
        engine = RegimeOverlayEngine(config)
        results = await engine.run()
        print(f"✅ Regime Overlay Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Regime Overlay Engine failed: {e}")
        return False, 0

async def run_ml_engine():
    """Run the ML engine"""
    print("\n" + "=" * 60)
    print("STEP 6: RUNNING ML ENGINE")
    print("=" * 60)
    
    try:
        from Engines.ml_engine import MLEngine, MLEngineConfig
        
        config = MLEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3,  # Limit for testing
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

async def run_performance_engine():
    """Run the performance engine"""
    print("\n" + "=" * 60)
    print("STEP 7: RUNNING PERFORMANCE ENGINE")
    print("=" * 60)
    
    try:
        from Engines.performance_engine import PerformanceEngine, PerformanceEngineConfig
        
        config = PerformanceEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3  # Limit for testing
        )
        
        engine = PerformanceEngine(config)
        results = await engine.run()
        print(f"✅ Performance Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Performance Engine failed: {e}")
        return False, 0

async def run_portfolio_engine():
    """Run the portfolio engine"""
    print("\n" + "=" * 60)
    print("STEP 8: RUNNING PORTFOLIO ENGINE")
    print("=" * 60)
    
    try:
        from Engines.portfolio_engine import PortfolioEngine, PortfolioEngineConfig
        
        config = PortfolioEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3  # Limit for testing
        )
        
        engine = PortfolioEngine(config)
        results = await engine.run()
        print(f"✅ Portfolio Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Portfolio Engine failed: {e}")
        return False, 0

async def run_validation_engine():
    """Run the validation engine"""
    print("\n" + "=" * 60)
    print("STEP 9: RUNNING VALIDATION ENGINE")
    print("=" * 60)
    
    try:
        from Engines.validation_engine import ValidationEngine, ValidationEngineConfig
        
        config = ValidationEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results",
            max_files=3  # Limit for testing
        )
        
        engine = ValidationEngine(config)
        results = await engine.run()
        print(f"✅ Validation Engine completed - {len(results)} results generated")
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Validation Engine failed: {e}")
        return False, 0

async def main():
    """Main function to run all 9 engines"""
    start_time = time.time()
    
    engines = [
        ("Core Engine", run_core_engine),
        ("Enhanced Risk Engine", run_enhanced_risk_engine),
        ("Enhanced Visualization Engine", run_enhanced_visualization_engine),
        ("Regime Detection Engine", run_regime_detection_engine),
        ("Regime Overlay Engine", run_regime_overlay_engine),
        ("ML Engine", run_ml_engine),
        ("Performance Engine", run_performance_engine),
        ("Portfolio Engine", run_portfolio_engine),
        ("Validation Engine", run_validation_engine)
    ]
    
    results = {}
    successful_engines = 0
    total_results = 0
    
    for engine_name, engine_func in engines:
        try:
            success, result_count = await engine_func()
            results[engine_name] = {"success": success, "results": result_count}
            if success:
                successful_engines += 1
                total_results += result_count
        except Exception as e:
            print(f"❌ {engine_name} failed with exception: {e}")
            results[engine_name] = {"success": False, "results": 0}
    
    execution_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL 9 ENGINES")
    print("=" * 80)
    
    for engine_name, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{engine_name}: {status} ({result['results']} results)")
    
    print(f"\nEngines Completed Successfully: {successful_engines}/9")
    print(f"Total Results Generated: {total_results}")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print(f"Results Location: C:\\Users\\andre\\OneDrive\\Desktop\\Mastercode\\Backtesting\\Results")
    
    if successful_engines < 9:
        print(f"\n⚠️  {9 - successful_engines} engines failed. Check the logs above for details.")
    else:
        print("\n🎉 All 9 engines completed successfully!")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
