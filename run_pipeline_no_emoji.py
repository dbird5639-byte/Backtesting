#!/usr/bin/env python3
"""
Simple Pipeline Runner - Runs engines in order with regime engines last (No Emojis)
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def run_core_engine():
    """Run Core Engine"""
    print("\n[1/12] Running Core Engine...")
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = CoreEngine(config)
        engine.run()
        
        print("SUCCESS: Core Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Core Engine failed: {e}")
        return False

def run_risk_engine():
    """Run Risk Engine"""
    print("\n[2/12] Running Risk Engine...")
    try:
        from Engines.risk_engine import RiskEngine, RiskEngineConfig
        
        config = RiskEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = RiskEngine(config)
        engine.run()
        
        print("SUCCESS: Risk Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Risk Engine failed: {e}")
        return False

def run_statistical_engine():
    """Run Statistical Engine"""
    print("\n[3/12] Running Statistical Engine...")
    try:
        from Engines.statistical_engine import StatisticalEngine, StatisticalEngineConfig
        
        config = StatisticalEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = StatisticalEngine(config)
        engine.run()
        
        print("SUCCESS: Statistical Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Statistical Engine failed: {e}")
        return False

def run_validation_engine():
    """Run Validation Engine"""
    print("\n[4/12] Running Validation Engine...")
    try:
        from Engines.validation_engine import ValidationEngine, ValidationEngineConfig
        
        config = ValidationEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = ValidationEngine(config)
        engine.run()
        
        print("SUCCESS: Validation Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Validation Engine failed: {e}")
        return False

def run_portfolio_engine():
    """Run Portfolio Engine"""
    print("\n[5/12] Running Portfolio Engine...")
    try:
        from Engines.portfolio_engine import PortfolioEngine, PortfolioEngineConfig
        
        config = PortfolioEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = PortfolioEngine(config)
        engine.run()
        
        print("SUCCESS: Portfolio Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Portfolio Engine failed: {e}")
        return False

def run_ml_engine():
    """Run ML Engine"""
    print("\n[6/12] Running ML Engine...")
    try:
        from Engines.ml_engine import MLEngine, MLEngineConfig
        
        config = MLEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = MLEngine(config)
        engine.run()
        
        print("SUCCESS: ML Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: ML Engine failed: {e}")
        return False

def run_performance_engine():
    """Run Performance Engine"""
    print("\n[7/12] Running Performance Engine...")
    try:
        from Engines.performance_engine import PerformanceEngine, PerformanceEngineConfig
        
        config = PerformanceEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = PerformanceEngine(config)
        engine.run()
        
        print("SUCCESS: Performance Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Performance Engine failed: {e}")
        return False

def run_visualization_engine():
    """Run Visualization Engine"""
    print("\n[8/12] Running Visualization Engine...")
    try:
        from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
        
        config = VisualizationConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = EnhancedVisualizationEngine(config)
        engine.run()
        
        print("SUCCESS: Visualization Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Visualization Engine failed: {e}")
        return False

def run_fibonacci_engine():
    """Run Fibonacci/Gann Engine"""
    print("\n[9/12] Running Fibonacci/Gann Engine...")
    try:
        from Engines.fibonacci_gann_engine import FibonacciGannEngine, FibonacciGannConfig
        
        config = FibonacciGannConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = FibonacciGannEngine(config)
        engine.run()
        
        print("SUCCESS: Fibonacci/Gann Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Fibonacci/Gann Engine failed: {e}")
        return False

def run_regime_detection_engine():
    """Run Regime Detection Engine (LAST)"""
    print("\n[10/12] Running Regime Detection Engine (LAST IN LINE)...")
    try:
        from Engines.regime_detection_engine import RegimeDetectionEngine, RegimeDetectionConfig
        
        config = RegimeDetectionConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = RegimeDetectionEngine(config)
        engine.run()
        
        print("SUCCESS: Regime Detection Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Regime Detection Engine failed: {e}")
        return False

def run_regime_overlay_engine():
    """Run Regime Overlay Engine (LAST)"""
    print("\n[11/12] Running Regime Overlay Engine (LAST IN LINE)...")
    try:
        from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
        
        config = RegimeOverlayConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = RegimeOverlayEngine(config)
        engine.run()
        
        print("SUCCESS: Regime Overlay Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Regime Overlay Engine failed: {e}")
        return False

def run_regime_visualization_engine():
    """Run Regime Visualization Engine (LAST)"""
    print("\n[12/12] Running Regime Visualization Engine (LAST IN LINE)...")
    try:
        from Engines.regime_visualization_engine import RegimeVisualizationEngine, RegimeVisualizationConfig
        
        config = RegimeVisualizationConfig()
        config.parallel_workers = 4
        config.verbose = True
        
        engine = RegimeVisualizationEngine(config)
        engine.run()
        
        print("SUCCESS: Regime Visualization Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Regime Visualization Engine failed: {e}")
        return False

def main():
    """Main pipeline runner"""
    start_time = time.time()
    
    print("=" * 80)
    print("STARTING OPTIMIZED PIPELINE RUNNER")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Define engine order with regime engines last
    engines = [
        ("Core Engine", run_core_engine),
        ("Risk Engine", run_risk_engine),
        ("Statistical Engine", run_statistical_engine),
        ("Validation Engine", run_validation_engine),
        ("Portfolio Engine", run_portfolio_engine),
        ("ML Engine", run_ml_engine),
        ("Performance Engine", run_performance_engine),
        ("Visualization Engine", run_visualization_engine),
        ("Fibonacci/Gann Engine", run_fibonacci_engine),
        # Regime engines LAST
        ("Regime Detection Engine", run_regime_detection_engine),
        ("Regime Overlay Engine", run_regime_overlay_engine),
        ("Regime Visualization Engine", run_regime_visualization_engine),
    ]
    
    results = {}
    
    # Run engines in order
    for i, (engine_name, engine_func) in enumerate(engines, 1):
        print(f"\nEngine {i}/{len(engines)}: {engine_name}")
        
        engine_start = time.time()
        success = engine_func()
        engine_time = time.time() - engine_start
        
        results[engine_name] = {
            'success': success,
            'execution_time': engine_time,
            'status': 'completed' if success else 'failed'
        }
        
        # Show progress
        completed = sum(1 for r in results.values() if r['success'])
        failed = sum(1 for r in results.values() if not r['success'])
        total_time = time.time() - start_time
        
        print(f"   Progress: {completed} completed, {failed} failed")
        print(f"   Engine time: {engine_time:.2f}s, Total time: {total_time:.2f}s")
        
        # Add separator for regime engines
        if "Regime" in engine_name and i == len(engines) - 2:  # First regime engine
            print("\n" + "=" * 60)
            print("REGIME ENGINES RUNNING (LAST IN LINE)")
            print("=" * 60)
    
    # Final summary
    total_time = time.time() - start_time
    completed = sum(1 for r in results.values() if r['success'])
    failed = sum(1 for r in results.values() if not r['success'])
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f}m)")
    print(f"Engines completed: {completed}/{len(engines)}")
    print(f"Engines failed: {failed}/{len(engines)}")
    print(f"Success rate: {completed/len(engines)*100:.1f}%")
    
    print("\nEngine Results:")
    for engine_name, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"   {status}: {engine_name}: {result['execution_time']:.2f}s - {result['status']}")
    
    # Regime engines summary
    regime_engines = [name for name in results.keys() if 'Regime' in name]
    if regime_engines:
        print(f"\nRegime Engines (Last in Line): {len(regime_engines)}")
        for engine_name in regime_engines:
            result = results[engine_name]
            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"   {status}: {engine_name}: {result['execution_time']:.2f}s")
    
    print("=" * 80)
    
    return completed == len(engines)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
