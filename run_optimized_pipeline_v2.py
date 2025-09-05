#!/usr/bin/env python3
"""
Optimized Pipeline Runner V2 - Uses optimized engines with integrated visualization
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def run_core_engine():
    """Run Core Engine"""
    print("\n[1/8] Running Core Engine...")
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
    print("\n[2/8] Running Risk Engine...")
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
    print("\n[3/8] Running Statistical Engine...")
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
    print("\n[4/8] Running Validation Engine...")
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

def run_optimized_portfolio_engine():
    """Run Optimized Portfolio Engine with Visualization"""
    print("\n[5/8] Running Optimized Portfolio Engine...")
    try:
        from Engines.optimized_portfolio_engine import OptimizedPortfolioEngine, PortfolioEngineConfig
        
        config = PortfolioEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        config.generate_plots = True
        
        engine = OptimizedPortfolioEngine(config)
        engine.run()
        
        print("SUCCESS: Optimized Portfolio Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Optimized Portfolio Engine failed: {e}")
        return False

def run_optimized_ml_engine():
    """Run Optimized ML Engine with Visualization"""
    print("\n[6/8] Running Optimized ML Engine...")
    try:
        from Engines.optimized_ml_engine import OptimizedMLEngine, MLEngineConfig
        
        config = MLEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        config.generate_plots = True
        
        engine = OptimizedMLEngine(config)
        engine.run()
        
        print("SUCCESS: Optimized ML Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Optimized ML Engine failed: {e}")
        return False

def run_performance_engine():
    """Run Performance Engine"""
    print("\n[7/8] Running Performance Engine...")
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

def run_optimized_regime_engine():
    """Run Optimized Combined Regime Engine (LAST)"""
    print("\n[8/8] Running Optimized Combined Regime Engine (LAST IN LINE)...")
    try:
        from Engines.optimized_regime_engine import OptimizedRegimeEngine, RegimeEngineConfig
        
        config = RegimeEngineConfig()
        config.parallel_workers = 4
        config.verbose = True
        config.generate_plots = True
        
        engine = OptimizedRegimeEngine(config)
        engine.run()
        
        print("SUCCESS: Optimized Combined Regime Engine completed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Optimized Combined Regime Engine failed: {e}")
        return False

def main():
    """Main pipeline runner"""
    start_time = time.time()
    
    print("=" * 80)
    print("STARTING OPTIMIZED PIPELINE RUNNER V2")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Features:")
    print("  - Optimized engines with integrated visualization")
    print("  - Combined regime engine (detection + overlay + visualization)")
    print("  - Enhanced portfolio and ML engines")
    print("  - Regime engines run LAST as requested")
    print("=" * 80)
    
    # Define engine order with regime engines last
    engines = [
        ("Core Engine", run_core_engine),
        ("Risk Engine", run_risk_engine),
        ("Statistical Engine", run_statistical_engine),
        ("Validation Engine", run_validation_engine),
        ("Optimized Portfolio Engine", run_optimized_portfolio_engine),
        ("Optimized ML Engine", run_optimized_ml_engine),
        ("Performance Engine", run_performance_engine),
        # Regime engines LAST
        ("Optimized Combined Regime Engine", run_optimized_regime_engine),
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
        if "Regime" in engine_name:
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
    
    print("\nOptimization Summary:")
    print("  - Portfolio Engine: Multi-objective optimization with visualizations")
    print("  - ML Engine: Advanced feature engineering with model visualizations")
    print("  - Regime Engine: Combined detection, overlay, and visualization")
    print("  - All engines: Integrated visualization, no separate visual engine needed")
    
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
