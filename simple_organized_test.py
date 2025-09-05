#!/usr/bin/env python3
"""
Simple test for organized results structure
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("Testing organized results structure...")
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        config = EngineConfig()
        config.parallel_workers = 1
        config.verbose = True
        config.results_path = "Results"
        
        engine = CoreEngine(config)
        
        # Get one data file and one strategy
        data_files = engine.discover_data_files()[:1]
        strategy_files = engine.discover_strategy_files()[:1]
        
        if not data_files or not strategy_files:
            print("No files found")
            return
        
        print(f"Testing: {data_files[0].name} + {strategy_files[0].name}")
        
        # Process one combination
        results = engine.process_file_strategy_combination(data_files[0], strategy_files[0])
        
        if results:
            print(f"Generated {len(results)} results")
            engine.save_results(results)
            print("Results saved with organized structure")
            
            # Check Results directory
            results_dir = Path("Results")
            if results_dir.exists():
                print("\nResults directory structure:")
                for item in results_dir.iterdir():
                    if item.is_dir():
                        print(f"  📁 {item.name}/")
                        for subitem in item.iterdir():
                            if subitem.is_dir():
                                print(f"    📁 {subitem.name}/")
                                for subsubitem in subitem.iterdir():
                                    if subsubitem.is_dir():
                                        print(f"      📁 {subsubitem.name}/")
                                        for file in subsubitem.iterdir():
                                            print(f"        📄 {file.name}")
                                    else:
                                        print(f"      📄 {subsubitem.name}")
                            else:
                                print(f"    📄 {subitem.name}")
        else:
            print("No results generated")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
