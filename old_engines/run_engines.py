#!/usr/bin/env python3
"""
Entry point script for running backtesting engines.

This script can be run directly to execute any of the backtesting engines
without import issues.
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
from epoch_engine import EpochEngine, EpochEngineConfig
from ml_engine import MLEngine, MLEngineConfig
from pipeline_engine import PipelineEngine, PipelineEngineConfig

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\strategies",
        r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\Data\winners",
        r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\results"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\nPlease create these directories before running the engines.")
        return False
    
    print("✅ All required directories exist")
    return True

def check_existing_results():
    """Check for existing results and show resume information"""
    results_dir = r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\results"
    
    if not os.path.exists(results_dir):
        print("📁 No existing results found")
        return
    
    # Find engine result directories
    engine_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and not item.startswith('logs'):
            engine_dirs.append(item)
    
    if not engine_dirs:
        print("📁 No engine result directories found")
        return
    
    print(f"\n📊 Found {len(engine_dirs)} existing result directories:")
    total_results = 0
    
    for engine_dir in sorted(engine_dirs):
        engine_path = os.path.join(results_dir, engine_dir)
        results_count = 0
        
        # Count results
        for strategy_dir in os.listdir(engine_path):
            strategy_path = os.path.join(engine_path, strategy_dir)
            if os.path.isdir(strategy_path):
                json_files = [f for f in os.listdir(strategy_path) if f.endswith('.json')]
                results_count += len(json_files)
        
        total_results += results_count
        
        if results_count > 0:
            print(f"  ✅ {engine_dir}: {results_count} results (can resume)")
        else:
            print(f"  ⚠️  {engine_dir}: No results (start fresh)")
    
    print(f"\n📈 Total existing results: {total_results}")
    
    if total_results > 0:
        print("\n💡 Tip: Engines will automatically skip already processed files")
        print("   Use 'resume_analysis.py' for detailed resume information")

def run_basic_engine():
    """Run the basic backtesting engine"""
    print("\n🚀 Running Basic Engine...")
    config = BasicEngineConfig(parallel_workers=2)
    engine = BasicEngine(config)
    return engine.run()

def run_risk_managed_engine():
    """Run the risk-managed backtesting engine"""
    print("\n🚀 Running Risk-Managed Engine...")
    config = RiskManagedEngineConfig(
        parallel_workers=2,
        backtest_timeout=43200,  # 12 hours
    )
    engine = RiskManagedEngine(config)
    return engine.run()

def run_risk_managed_engine_oos():
    """Run risk-managed engine with OOS-only selection for walkforward params"""
    print("\n🚀 Running Risk-Managed Engine (OOS selection)...")
    config = RiskManagedEngineConfig(
        parallel_workers=2,
        backtest_timeout=43200,
        walkforward_selection_criterion="oos",
        walkforward_window_mode="rolling",
    )
    engine = RiskManagedEngine(config)
    return engine.run()

def run_risk_managed_engine_is():
    """Run risk-managed engine with IS-only selection for walkforward params"""
    print("\n🚀 Running Risk-Managed Engine (IS selection)...")
    config = RiskManagedEngineConfig(
        parallel_workers=2,
        backtest_timeout=43200,
        walkforward_selection_criterion="is",
        walkforward_window_mode="rolling",
    )
    engine = RiskManagedEngine(config)
    return engine.run()

def run_risk_managed_engine_combined():
    """Run risk-managed engine with combined selection for walkforward params"""
    print("\n🚀 Running Risk-Managed Engine (Combined selection)...")
    config = RiskManagedEngineConfig(
        parallel_workers=2,
        backtest_timeout=43200,
        walkforward_selection_criterion="combined",
        walkforward_window_mode="rolling",
    )
    engine = RiskManagedEngine(config)
    return engine.run()

def run_risk_managed_engine_all_selection_modes():
    """Run risk-managed engine for IS, OOS, and Combined selection sequentially"""
    print("\n🔁 Running Risk-Managed Engine for all selection modes (IS, OOS, Combined)...")
    run_risk_managed_engine_is()
    run_risk_managed_engine_oos()
    run_risk_managed_engine_combined()
    print("\n✅ Completed all selection modes.")

def run_statistical_engine():
    """Run the statistical backtesting engine"""
    print("\n🚀 Running Statistical Engine...")
    config = StatisticalEngineConfig(parallel_workers=2)
    engine = StatisticalEngine(config)
    return engine.run()

def run_walkforward_engine():
    """Run the walkforward backtesting engine"""
    print("\n🚀 Running Walkforward Engine...")
    # Allow overriding min_total_size via environment for flexibility
    try:
        _min_total = int(os.getenv('WALKFORWARD_MIN_TOTAL_SIZE', '0'))
    except Exception:
        _min_total = 0
    if _min_total and _min_total > 0:
        # If requested min_total is smaller than default train+test, scale train/test to fit
        default_train = 1000
        default_test = 200
        required_total = default_train + default_test
        if _min_total < required_total:
            # Keep ~80/20 split between train and test
            train_size = max(50, int(_min_total * 0.8))
            test_size = max(10, _min_total - train_size)
            step_size = max(1, min(test_size, 200))
            config = WalkforwardConfig(
                parallel_workers=2,
                min_total_size=_min_total,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
            )
        else:
            config = WalkforwardConfig(parallel_workers=2, min_total_size=_min_total)
    else:
        config = WalkforwardConfig(parallel_workers=2)
    engine = WalkforwardEngine(config)
    return engine.run()

def run_alpha_engine():
    """Run the alpha detection engine"""
    print("\n🚀 Running Alpha Engine...")
    config = AlphaEngineConfig()
    engine = AlphaEngine(config)
    return engine.run()

def run_pipeline_engine():
    """Run the pipeline engine"""
    print("\n🚀 Running Pipeline Engine...")
    config = PipelineEngineConfig()
    engine = PipelineEngine(config)
    return engine.run()

def run_targeted_pipeline():
    """Run a targeted pipeline with specific engines"""
    print("\n🚀 Running Targeted Pipeline...")
    config = PipelineEngineConfig(
        engines_to_run=['basic', 'risk_managed', 'walkforward'],
        enable_cross_engine_analysis=True
    )
    engine = PipelineEngine(config)
    return engine.run()

def main():
    """Main function to run the selected engine"""
    print("🔧 Backtesting Engines Runner")
    print("=" * 50)
    
    # Check directories first
    if not check_directories():
        return
    
    # Check existing results
    check_existing_results()
    
    # Show available options
    print("\n📋 Available Engines:")
    print("1. Basic Engine")
    print("2. Risk-Managed Engine")
    print("3. Statistical Engine")
    print("4. Walkforward Engine")
    print("5. Alpha Engine")
    print("6. Pipeline Engine (All engines)")
    print("7. Epoch Engine")
    print("8. ML Engine")
    print("9. Targeted Pipeline (Basic + Risk + Walkforward)")
    print("10. Run All Engines Sequentially")
    print("11. Resume Analysis (Show detailed resume info)")
    
    try:
        choice = input("\nSelect an engine to run (1-9): ").strip()
        
        if choice == "1":
            run_basic_engine()
        elif choice == "2":
            run_risk_managed_engine()
        elif choice == "3":
            run_statistical_engine()
        elif choice == "4":
            run_walkforward_engine()
        elif choice == "5":
            run_alpha_engine()
        elif choice == "6":
            run_pipeline_engine()
        elif choice == "7":
            print("\n🚀 Running Epoch Engine...")
            engine = EpochEngine(EpochEngineConfig(parallel_workers=2))
            engine.run()
        elif choice == "8":
            print("\n🚀 Running ML Engine...")
            engine = MLEngine(MLEngineConfig(parallel_workers=2))
            engine.run()
        elif choice == "9":
            run_targeted_pipeline()
        elif choice == "10":
            print("\n🔄 Running all engines sequentially...")
            run_basic_engine()
            run_risk_managed_engine()
            run_statistical_engine()
            run_walkforward_engine()
            run_alpha_engine()
            # Optional extras
            EpochEngine(EpochEngineConfig(parallel_workers=2)).run()
            MLEngine(MLEngineConfig(parallel_workers=2)).run()
            print("\n✅ All engines completed!")
        elif choice == "11":
            print("\n📊 Running resume analysis...")
            import subprocess
            subprocess.run([sys.executable, "resume_analysis.py"])
        else:
            print("❌ Invalid choice. Please select 1-9.")
            return
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running engine: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 