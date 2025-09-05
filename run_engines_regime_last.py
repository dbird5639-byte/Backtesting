#!/usr/bin/env python3
"""
Run Engines with Regime Engines Last
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("RUNNING ENGINES WITH REGIME ENGINES LAST")
print("=" * 80)

async def run_core_engine():
    print("\nSTEP 1: RUNNING CORE ENGINE")
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        config = EngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            save_json=False, save_csv=True, save_plots=False, max_data_points=1000
        )
        engine = CoreEngine(config)
        results = await engine.run()
        print("✅ Core Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Core Engine failed: {e}")
        return False, 0

async def run_enhanced_risk_engine():
    print("\nSTEP 2: RUNNING ENHANCED RISK ENGINE")
    try:
        from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig
        config = EnhancedRiskEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        engine = EnhancedRiskEngine(config)
        results = await engine.run()
        print("✅ Enhanced Risk Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Enhanced Risk Engine failed: {e}")
        return False, 0

async def run_ml_engine():
    print("\nSTEP 3: RUNNING ML ENGINE")
    try:
        from Engines.ml_engine import MLEngine, MLEngineConfig
        config = MLEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        engine = MLEngine(config)
        results = await engine.run()
        print("✅ ML Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ ML Engine failed: {e}")
        return False, 0

async def run_performance_engine():
    print("\nSTEP 4: RUNNING PERFORMANCE ENGINE")
    try:
        from Engines.performance_engine import PerformanceEngine, PerformanceEngineConfig
        config = PerformanceEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        engine = PerformanceEngine(config)
        results = await engine.run()
        print("✅ Performance Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Performance Engine failed: {e}")
        return False, 0

async def run_portfolio_engine():
    print("\nSTEP 5: RUNNING PORTFOLIO ENGINE")
    try:
        from Engines.portfolio_engine import PortfolioEngine, PortfolioEngineConfig
        config = PortfolioEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        engine = PortfolioEngine(config)
        results = await engine.run()
        print("✅ Portfolio Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Portfolio Engine failed: {e}")
        return False, 0

async def run_validation_engine():
    print("\nSTEP 6: RUNNING VALIDATION ENGINE")
    try:
        from Engines.validation_engine import ValidationEngine, ValidationEngineConfig
        config = ValidationEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        engine = ValidationEngine(config)
        results = await engine.run()
        print("✅ Validation Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Validation Engine failed: {e}")
        return False, 0

async def run_enhanced_visualization_engine():
    print("\nSTEP 7: RUNNING ENHANCED VISUALIZATION ENGINE")
    try:
        from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
        config = VisualizationConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        engine = EnhancedVisualizationEngine(config)
        results = await engine.run()
        print("✅ Enhanced Visualization Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Enhanced Visualization Engine failed: {e}")
        return False, 0

async def run_regime_detection_engine():
    print("\nSTEP 8: RUNNING REGIME DETECTION ENGINE (LAST)")
    try:
        from Engines.regime_detection_engine import RegimeDetectionEngine
        config = {
            'data_path': r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            'results_path': r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            'n_regimes': 3
        }
        engine = RegimeDetectionEngine(config)
        results = await engine.run()
        print("✅ Regime Detection Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Regime Detection Engine failed: {e}")
        return False, 0

async def run_regime_overlay_engine():
    print("\nSTEP 9: RUNNING REGIME OVERLAY ENGINE (LAST)")
    try:
        from Engines.regime_overlay_engine import RegimeOverlayEngine
        config = {
            'data_path': r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            'results_path': r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        }
        engine = RegimeOverlayEngine(config)
        results = await engine.run()
        print("✅ Regime Overlay Engine completed")
        return True, len(results)
    except Exception as e:
        print(f"❌ Regime Overlay Engine failed: {e}")
        return False, 0

async def main():
    engines = [
        ("Core Engine", run_core_engine),
        ("Enhanced Risk Engine", run_enhanced_risk_engine),
        ("ML Engine", run_ml_engine),
        ("Performance Engine", run_performance_engine),
        ("Portfolio Engine", run_portfolio_engine),
        ("Validation Engine", run_validation_engine),
        ("Enhanced Visualization Engine", run_enhanced_visualization_engine),
        ("Regime Detection Engine (LAST)", run_regime_detection_engine),
        ("Regime Overlay Engine (LAST)", run_regime_overlay_engine)
    ]
    
    results = {}
    successful = 0
    
    for name, func in engines:
        success, count = await func()
        results[name] = {"success": success, "results": count}
        if success:
            successful += 1
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - REGIME ENGINES LAST")
    print(f"{'='*60}")
    for name, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{name}: {status}")
    print(f"\nCompleted: {successful}/9 engines")

if __name__ == "__main__":
    asyncio.run(main())
