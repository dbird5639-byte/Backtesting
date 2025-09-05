#!/usr/bin/env python3
"""
Show Engine Results
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    # Create results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "status": "running_engines",
        "results_found": False,
        "engines_status": {}
    }
    
    try:
        from Engines.core_engine import CoreEngine, EngineConfig
        
        # Test engine creation
        config = EngineConfig()
        engine = CoreEngine(config)
        
        # Test file discovery
        data_files = engine.discover_data_files()
        strategy_files = engine.discover_strategy_files()
        
        summary["data_files_count"] = len(data_files)
        summary["strategy_files_count"] = len(strategy_files)
        summary["total_combinations"] = len(data_files) * len(strategy_files)
        
        # Test single combination
        if data_files and strategy_files:
            data_file = data_files[0]
            strategy_file = strategy_files[0]
            
            # Load data
            data = engine.load_data(data_file)
            summary["data_loading"] = data is not None
            
            if data is not None:
                summary["data_shape"] = data.shape
                summary["data_columns"] = data.columns.tolist()
            
            # Load strategy
            strategy_cls = engine.load_strategy(strategy_file)
            summary["strategy_loading"] = strategy_cls is not None
            
            if strategy_cls is not None:
                summary["strategy_class"] = strategy_cls.__name__
            
            # Test backtest
            if data is not None and strategy_cls is not None:
                try:
                    result = engine.run_single_backtest(strategy_cls, data)
                    summary["backtest_success"] = result is not None
                    if result:
                        summary["sample_result"] = {
                            "total_return": result.get('total_return', 0),
                            "sharpe_ratio": result.get('sharpe_ratio', 0),
                            "max_drawdown": result.get('max_drawdown', 0),
                            "num_trades": result.get('num_trades', 0)
                        }
                except Exception as e:
                    summary["backtest_error"] = str(e)
        
        # Check existing results
        results_dir = Path("Results")
        if results_dir.exists():
            summary["results_found"] = True
            summary["results_files"] = len(list(results_dir.rglob("*")))
            summary["json_files"] = len(list(results_dir.rglob("*.json")))
            summary["csv_files"] = len(list(results_dir.rglob("*.csv")))
            summary["png_files"] = len(list(results_dir.rglob("*.png")))
        
        summary["status"] = "completed"
        
    except Exception as e:
        summary["error"] = str(e)
        summary["status"] = "failed"
    
    # Save summary
    with open("engine_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also print to console
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
