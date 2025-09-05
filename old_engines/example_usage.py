#!/usr/bin/env python3
"""
Example usage of the backtesting engines.

This script demonstrates how to use each engine individually and in a pipeline.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import the engines
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now we can import the engines
from base_engine import EngineConfig
from basic_engine import BasicEngine, BasicEngineConfig
from risk_managed_engine import RiskManagedEngine, RiskManagedEngineConfig
from statistical_engine import StatisticalEngine, StatisticalEngineConfig
from walkforward_engine import WalkforwardEngine, WalkforwardConfig
from alpha_engine import AlphaEngine, AlphaEngineConfig
from pipeline_engine import PipelineEngine, PipelineEngineConfig

def check_requirements():
    """Check if required directories and files exist"""
    required_dirs = [
        r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\strategies",
        r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\Data\Hyperliquid",
        r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\results"
    ]
    
    print("🔍 Checking requirements...")
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            return False
    
    # Check if there are any strategies
    strategies_dir = required_dirs[0]
    strategies = [f for f in os.listdir(strategies_dir) if f.endswith('.py')] if os.path.exists(strategies_dir) else []
    if strategies:
        print(f"✅ Found {len(strategies)} strategies")
    else:
        print("⚠️  No strategies found")
    
    # Check if there are any data files
    data_dir = required_dirs[1]
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')] if os.path.exists(data_dir) else []
    if data_files:
        print(f"✅ Found {len(data_files)} data files")
    else:
        print("⚠️  No data files found")
    
    return True

def run_basic_engine_example():
    """Example of running the basic engine"""
    print("\n🚀 Running Basic Engine Example...")
    
    # Create configuration
    config = BasicEngineConfig(
        initial_cash=10000.0,
        commission=0.002,
        backtest_timeout=60,
        save_json=True,
        save_csv=False,
        save_plots=False
    )
    
    # Create and run engine
    engine = BasicEngine(config)
    results = engine.run()
    
    print(f"✅ Basic engine completed with {len(results) if results else 0} results")
    return results

def run_risk_managed_engine_example():
    """Example of running the risk-managed engine"""
    print("\n🚀 Running Risk-Managed Engine Example...")
    
    # Create configuration with walkforward optimization
    config = RiskManagedEngineConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        position_size_pct=0.10,
        enable_walkforward_optimization=True,
        walkforward_train_size=500,
        walkforward_test_size=100
    )
    
    # Create and run engine
    engine = RiskManagedEngine(config)
    results = engine.run()
    
    print(f"✅ Risk-managed engine completed with {len(results) if results else 0} results")
    return results

def run_statistical_engine_example():
    """Example of running the statistical engine"""
    print("\n🚀 Running Statistical Engine Example...")
    
    # Create configuration
    config = StatisticalEngineConfig(
        n_permutations=100,
        n_bootstrap_samples=1000,
        n_monte_carlo_simulations=1000,
        n_regimes=3
    )
    
    # Create and run engine
    engine = StatisticalEngine(config)
    results = engine.run()
    
    print(f"✅ Statistical engine completed with {len(results) if results else 0} results")
    return results

def run_walkforward_engine_example():
    """Example of running the walkforward engine"""
    print("\n🚀 Running Walkforward Engine Example...")
    
    # Create configuration for performance analysis only
    config = WalkforwardConfig(
        train_size=1000,
        test_size=200,
        step_size=200,
        enable_regime_analysis=True,
        n_regimes=3
    )
    
    # Create and run engine
    engine = WalkforwardEngine(config)
    results = engine.run()
    
    print(f"✅ Walkforward engine completed with {len(results) if results else 0} results")
    return results

def run_alpha_engine_example():
    """Example of running the alpha engine"""
    print("\n🚀 Running Alpha Engine Example...")
    
    # Create configuration
    config = AlphaEngineConfig(
        alpha_periods=[5, 10, 20, 50],
        decay_threshold=0.1,
        signal_strength_threshold=0.5
    )
    
    # Create and run engine
    engine = AlphaEngine(config)
    results = engine.run()
    
    print(f"✅ Alpha engine completed with {len(results) if results else 0} results")
    return results

def run_pipeline_engine_example():
    """Example of running the pipeline engine"""
    print("\n🚀 Running Pipeline Engine Example...")
    
    # Create configuration to run all engines
    config = PipelineEngineConfig(
        engines_to_run=['basic', 'risk_managed', 'statistical', 'walkforward', 'alpha'],
        enable_cross_engine_analysis=True,
        save_combined_results=True
    )
    
    # Create and run engine
    engine = PipelineEngine(config)
    results = engine.run()
    
    print(f"✅ Pipeline engine completed with {len(results) if results else 0} results")
    return results

def run_targeted_pipeline_example():
    """Example of running a targeted pipeline"""
    print("\n🚀 Running Targeted Pipeline Example...")
    
    # Create configuration for specific engines
    config = PipelineEngineConfig(
        engines_to_run=['basic', 'risk_managed', 'walkforward'],
        enable_cross_engine_analysis=True,
        save_combined_results=True
    )
    
    # Create and run engine
    engine = PipelineEngine(config)
    results = engine.run()
    
    print(f"✅ Targeted pipeline completed with {len(results) if results else 0} results")
    return results

def main():
    """Main function to run examples"""
    print("🔧 Backtesting Engines - Example Usage")
    print("=" * 50)
    
    # Check requirements first
    if not check_requirements():
        print("\n❌ Requirements not met. Please ensure all directories exist and contain data.")
        return
    
    print("\n📋 Available Examples:")
    print("1. Basic Engine")
    print("2. Risk-Managed Engine (with walkforward optimization)")
    print("3. Statistical Engine")
    print("4. Walkforward Engine (performance analysis only)")
    print("5. Alpha Engine")
    print("6. Pipeline Engine (all engines)")
    print("7. Targeted Pipeline (Basic + Risk + Walkforward)")
    print("8. Run All Examples")
    
    try:
        choice = input("\nSelect an example to run (1-8): ").strip()
        
        if choice == "1":
            run_basic_engine_example()
        elif choice == "2":
            run_risk_managed_engine_example()
        elif choice == "3":
            run_statistical_engine_example()
        elif choice == "4":
            run_walkforward_engine_example()
        elif choice == "5":
            run_alpha_engine_example()
        elif choice == "6":
            run_pipeline_engine_example()
        elif choice == "7":
            run_targeted_pipeline_example()
        elif choice == "8":
            print("\n🔄 Running all examples...")
            run_basic_engine_example()
            run_risk_managed_engine_example()
            run_statistical_engine_example()
            run_walkforward_engine_example()
            run_alpha_engine_example()
            run_pipeline_engine_example()
            print("\n✅ All examples completed!")
        else:
            print("❌ Invalid choice. Please select 1-8.")
            return
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 