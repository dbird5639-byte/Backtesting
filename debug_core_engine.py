#!/usr/bin/env python3
"""
Debug Core Engine
"""

import sys
from pathlib import Path

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("Debug Core Engine...")
    
    try:
        print("1. Importing...")
        from Engines.core_engine import CoreEngine, EngineConfig
        print("✅ Import successful")
        
        print("2. Creating config...")
        config = EngineConfig()
        config.parallel_workers = 2
        config.verbose = True
        config.results_path = "Results"
        print("✅ Config created")
        
        print("3. Creating engine...")
        engine = CoreEngine(config)
        print("✅ Engine created")
        
        print("4. Discovering files...")
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        print(f"✅ Found {len(data_files)} data files, {len(strategy_files)} strategy files")
        
        if len(data_files) > 0 and len(strategy_files) > 0:
            print("5. Testing single combination...")
            result = engine.process_file_strategy_combination(data_files[0], strategy_files[0])
            print(f"✅ Processed combination, got {len(result)} results")
            
            if result:
                print("6. Testing save results...")
                engine.save_results(result)
                print("✅ Results saved")
                
                # Check Results directory
                results_dir = Path("Results")
                if results_dir.exists():
                    print("7. Results directory structure:")
                    for item in results_dir.iterdir():
                        if item.is_dir():
                            print(f"  📁 {item.name}/")
                            for subitem in item.iterdir():
                                if subitem.is_dir():
                                    print(f"    📁 {subitem.name}/")
                                    for subsubitem in subitem.iterdir():
                                        if subsubitem.is_dir():
                                            print(f"      📁 {subsubitem.name}/")
                                            file_count = len(list(subsubitem.iterdir()))
                                            print(f"        📄 {file_count} files")
                                        else:
                                            print(f"      📄 {subsubitem.name}")
                                else:
                                    print(f"    📄 {subitem.name}")
                        else:
                            print(f"  📄 {item.name}")
        else:
            print("❌ No files found")
        
        print("✅ Debug completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
