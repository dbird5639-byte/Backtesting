import os
import shutil
import json
from pathlib import Path

def organize_winners():
    # Define paths
    base_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting")
    hyperliquid_path = base_path / "Data" / "Hyperliquid"
    winners_path = base_path / "Data" / "winners"
    results_path = base_path / "results" / "basic_backtest_20250802_204712"
    
    # Get list of available files in Hyperliquid
    available_files = set()
    for file in hyperliquid_path.glob("*.csv"):
        available_files.add(file.name)
    
    print(f"Found {len(available_files)} files in Hyperliquid folder")
    
    # Strategy folders to process
    strategies = [
        "bollinger_bands_strategy",
        "fibonacci_retracement_strategy", 
        "flux_vwap_pivot_strategy",
        "macd_strategy",
        "rsi_strategy",
        "vwap_strategy",
        "stochastic_oscillator_strategy",
        "parabolic_sar_strategy",
        "mean_reversion_zscore",
        "mean_reversion_rsi",
        "mean_reversion_bollinger",
        "mean_reversion_backtest",
        "sunny_bands_strategy",
        "simple_grid_scalping",
        "silver_bullet_am_session_backtest",
        "volume_breakout_backtest",
        "volume_distribution_backtest",
        "wma_crossover_trend_strategy",
        "moving_average_crossover"
    ]
    
    # Process each strategy
    for strategy in strategies:
        strategy_results_path = results_path / strategy / "winners"
        strategy_winners_path = winners_path / strategy
        
        if not strategy_results_path.exists():
            print(f"No results folder for {strategy}")
            continue
            
        if not strategy_winners_path.exists():
            print(f"Creating winners folder for {strategy}")
            strategy_winners_path.mkdir(exist_ok=True)
        
        # Get winning files from results
        winning_files = []
        for file in strategy_results_path.glob("*.json"):
            # Extract the base filename from the JSON result
            # The JSON filename should correspond to the CSV data file
            base_name = file.stem
            csv_name = f"{base_name}.csv"
            winning_files.append(csv_name)
        
        print(f"\n{strategy}: {len(winning_files)} winning files")
        
        # Copy available files
        copied_count = 0
        missing_count = 0
        
        for winning_file in winning_files:
            source_path = hyperliquid_path / winning_file
            
            if source_path.exists():
                dest_path = strategy_winners_path / winning_file
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"  ✓ Copied: {winning_file}")
                    copied_count += 1
                except Exception as e:
                    print(f"  ✗ Error copying {winning_file}: {e}")
            else:
                print(f"  ✗ Missing: {winning_file}")
                missing_count += 1
        
        print(f"  Copied: {copied_count}, Missing: {missing_count}")
    
    print(f"\nOrganization complete!")

if __name__ == "__main__":
    organize_winners()
