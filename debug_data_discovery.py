#!/usr/bin/env python3
"""
Debug Data Discovery
This script checks what data files and strategies are being discovered.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def debug_data_discovery():
    """Debug data and strategy discovery"""
    print("Debugging Data and Strategy Discovery")
    print("=" * 50)
    
    try:
        from Engines.core_engine import DataLoader, StrategyLoader, EngineConfig
        
        # Create config with correct paths
        config = EngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        )
        
        print(f"Data Path: {config.data_path}")
        print(f"Strategies Path: {config.strategies_path}")
        print(f"Results Path: {config.results_path}")
        
        # Check if paths exist
        data_path = Path(config.data_path)
        strategies_path = Path(config.strategies_path)
        
        print(f"\nData Path Exists: {data_path.exists()}")
        print(f"Strategies Path Exists: {strategies_path.exists()}")
        
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            print(f"CSV Files Found: {len(csv_files)}")
            if len(csv_files) <= 10:
                for file in csv_files:
                    print(f"  - {file.name}")
            else:
                print(f"  First 10 files:")
                for file in csv_files[:10]:
                    print(f"    - {file.name}")
                print(f"  ... and {len(csv_files) - 10} more")
        
        if strategies_path.exists():
            strategy_files = list(strategies_path.glob("*.py"))
            print(f"\nStrategy Files Found: {len(strategy_files)}")
            for file in strategy_files:
                print(f"  - {file.name}")
        
        # Test data loader
        print("\n" + "=" * 50)
        print("Testing DataLoader...")
        data_loader = DataLoader(config)
        data_files = await data_loader.discover_data_files()
        print(f"DataLoader found {len(data_files)} files")
        for file in data_files[:5]:  # Show first 5
            print(f"  - {file}")
        
        # Test strategy loader
        print("\n" + "=" * 50)
        print("Testing StrategyLoader...")
        strategy_loader = StrategyLoader(config)
        strategy_files = await strategy_loader.discover_strategies()
        print(f"StrategyLoader found {len(strategy_files)} files")
        for file in strategy_files:
            print(f"  - {file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_data_discovery())
