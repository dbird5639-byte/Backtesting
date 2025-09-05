#!/usr/bin/env python3
import sys
from pathlib import Path
import json
from datetime import datetime

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    with open("execution_log.txt", "w") as log:
        log.write(f"=== EXECUTION LOG - {datetime.now()} ===\n")
        
        try:
            log.write("1. Importing CoreEngine...\n")
            from Engines.core_engine import CoreEngine, EngineConfig
            
            log.write("2. Creating config...\n")
            config = EngineConfig()
            config.parallel_workers = 1
            config.verbose = True
            config.results_path = "Results"
            
            log.write("3. Creating engine...\n")
            engine = CoreEngine(config)
            
            log.write("4. Discovering files...\n")
            data_files = engine.discover_data_files()
            strategy_files = engine.discover_strategy_files()
            
            log.write(f"Found {len(data_files)} data files\n")
            log.write(f"Found {len(strategy_files)} strategy files\n")
            
            if not data_files or not strategy_files:
                log.write("ERROR: No files found!\n")
                return
            
            log.write("5. Running engine...\n")
            engine.run()
            
            log.write("6. Checking results...\n")
            results_dir = Path("Results")
            if results_dir.exists():
                all_files = list(results_dir.rglob("*"))
                log.write(f"Results directory has {len(all_files)} files\n")
                
                json_files = list(results_dir.rglob("*.json"))
                log.write(f"Found {len(json_files)} JSON files\n")
                
                csv_files = list(results_dir.rglob("*.csv"))
                log.write(f"Found {len(csv_files)} CSV files\n")
                
                png_files = list(results_dir.rglob("*.png"))
                log.write(f"Found {len(png_files)} PNG files\n")
            else:
                log.write("No Results directory found\n")
            
            log.write("=== EXECUTION COMPLETED ===\n")
            
        except Exception as e:
            log.write(f"ERROR: {e}\n")
            import traceback
            log.write(traceback.format_exc())

if __name__ == "__main__":
    main()
