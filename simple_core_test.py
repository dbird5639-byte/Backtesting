#!/usr/bin/env python3
"""
Simple Core Engine Test
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("Starting simple core engine test...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig()
        config.parallel_workers = 1
        config.verbose = True
        config.results_path = "Results"
        
        engine = CoreEngine(config)
        
        # Get files
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        print(f"Data files: {len(data_files)}")
        print(f"Strategy files: {len(strategy_files)}")
        
        if len(data_files) > 0 and len(strategy_files) > 0:
            print("Processing first combination...")
            result = engine.process_file_strategy_combination(data_files[0], strategy_files[0])
            print(f"Got {len(result)} results")
            
            if result:
                print("Saving results...")
                engine.save_results(result)
                print("Results saved")
                
                # Check Results directory
                results_dir = Path("Results")
                if results_dir.exists():
                    print("Results directory contents:")
                    for item in results_dir.iterdir():
                        print(f"  {item.name}")
                        if item.is_dir():
                            for subitem in item.iterdir():
                                print(f"    {subitem.name}")
        else:
            print("No files found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
