#!/usr/bin/env python3
"""
Test Organized Results Structure
Demonstrates the new Results/Engine(1,2,3,etc)/Strategy(1,2,3,etc)/ structure
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def test_core_engine_organized():
    """Test Core Engine with organized results"""
    print("=" * 80)
    print("TESTING ORGANIZED RESULTS STRUCTURE")
    print("=" * 80)
    print("Structure: Results/Engine(1,2,3,etc)/Strategy(1,2,3,etc)/Data_file/")
    print("Formats: .csv, .json, .png, engine-specific formats")
    print("=" * 80)
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        print("\n[1/1] Testing Core Engine with organized results...")
        
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        config.results_path = "Results"  # Use the Results directory
        
        engine = CoreEngine(config)
        
        # Run with limited data for testing
        data_files = engine.discover_data_files()[:2]  # Only first 2 data files
        strategy_files = engine.discover_strategy_files()[:2]  # Only first 2 strategies
        
        if not data_files or not strategy_files:
            print("ERROR: No data files or strategy files found")
            return False
        
        print(f"Testing with {len(data_files)} data files and {len(strategy_files)} strategies")
        
        # Process combinations
        all_results = []
        for data_file in data_files:
            for strategy_file in strategy_files:
                print(f"Processing: {data_file.name} + {strategy_file.name}")
                try:
                    results = engine.process_file_strategy_combination(data_file, strategy_file)
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing {data_file.name} + {strategy_file.name}: {e}")
        
        # Save results with new organized structure
        if all_results:
            engine.save_results(all_results)
            print(f"\nSUCCESS: {len(all_results)} results saved with organized structure")
            
            # Show the directory structure
            results_dir = Path("Results")
            if results_dir.exists():
                print("\nDirectory structure created:")
                self.show_directory_structure(results_dir, max_depth=4)
            
            return True
        else:
            print("ERROR: No results generated")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_directory_structure(directory: Path, max_depth: int = 3, current_depth: int = 0):
    """Show directory structure"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(directory.iterdir())
        for item in items:
            indent = "  " * current_depth
            if item.is_dir():
                print(f"{indent}📁 {item.name}/")
                show_directory_structure(item, max_depth, current_depth + 1)
            else:
                file_size = item.stat().st_size
                size_str = f"({file_size} bytes)" if file_size < 1024 else f"({file_size/1024:.1f} KB)"
                print(f"{indent}📄 {item.name} {size_str}")
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")

def main():
    """Main test function"""
    start_time = time.time()
    
    success = test_core_engine_organized()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    if success:
        print("✅ ORGANIZED RESULTS TEST COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print("\nResults are now organized as:")
        print("  Results/")
        print("  └── Engine_CoreEngine_YYYYMMDD_HHMMSS/")
        print("      ├── Strategy_strategy1/")
        print("      │   ├── Data_datafile1/")
        print("      │   │   ├── Data_datafile1_result.json")
        print("      │   │   ├── Data_datafile1_metrics.csv")
        print("      │   │   └── visualizations/")
        print("      │   │       └── *.png files")
        print("      │   ├── Data_datafile2/")
        print("      │   │   └── ...")
        print("      │   └── strategy_summary.json")
        print("      └── Strategy_strategy2/")
        print("          └── ...")
    else:
        print("❌ ORGANIZED RESULTS TEST FAILED!")
    
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
