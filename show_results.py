#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import os

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

def main():
    print("=== BACKTESTING RESULTS ===")
    
    # Check if Results directory exists
    results_dir = Path("Results")
    if not results_dir.exists():
        print("❌ No Results directory found")
        return
    
    # List all results
    print(f"\n📁 Results directory contents:")
    for item in results_dir.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # Check for engine results
            for subitem in item.iterdir():
                if subitem.is_dir():
                    print(f"    📁 {subitem.name}/")
                    file_count = len(list(subitem.iterdir()))
                    print(f"      📄 {file_count} files")
                else:
                    print(f"    📄 {subitem.name}")
        else:
            print(f"  📄 {item.name}")
    
    # Check for any JSON files with results
    json_files = list(results_dir.rglob("*.json"))
    if json_files:
        print(f"\n📊 Found {len(json_files)} JSON result files:")
        for json_file in json_files:
            print(f"  📄 {json_file.relative_to(results_dir)}")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        print(f"    ✅ {len(data)} results")
                        if 'total_return' in data[0]:
                            returns = [r.get('total_return', 0) for r in data]
                            avg_return = sum(returns) / len(returns)
                            print(f"    📈 Average return: {avg_return:.2%}")
                    elif isinstance(data, dict):
                        print(f"    📊 Summary data")
            except Exception as e:
                print(f"    ❌ Error reading: {e}")
    
    # Check for CSV files
    csv_files = list(results_dir.rglob("*.csv"))
    if csv_files:
        print(f"\n📈 Found {len(csv_files)} CSV files")
    
    # Check for PNG files (charts)
    png_files = list(results_dir.rglob("*.png"))
    if png_files:
        print(f"\n📊 Found {len(png_files)} chart files")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total files: {len(list(results_dir.rglob('*')))}")
    print(f"JSON files: {len(json_files)}")
    print(f"CSV files: {len(csv_files)}")
    print(f"PNG files: {len(png_files)}")

if __name__ == "__main__":
    main()
