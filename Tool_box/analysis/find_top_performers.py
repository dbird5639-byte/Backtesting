import os
import json

# Directories to search
result_dirs = [
    "results/basic_backtest_20250713_193232",
    "results/walkforward_20250714_004129"
]
fish_types = ["small_fish", "mid_fish", "big_fish"]

# Function to score a result (tuple for sorting)
def score_result(data):
    # Use Sharpe, then return_pct, then profit_factor as tiebreakers
    sharpe = data.get("sharpe_ratio", float('-inf'))
    ret = data.get("return_pct", float('-inf'))
    pf = data.get("profit_factor", float('-inf'))
    return (sharpe, ret, pf)

for result_dir in result_dirs:
    print(f"\nAnalyzing: {result_dir}")
    for strat in os.listdir(result_dir):
        strat_path = os.path.join(result_dir, strat)
        if not os.path.isdir(strat_path) or strat == "winners":
            continue
        winners_path = os.path.join(strat_path, "winners")
        if not os.path.isdir(winners_path):
            continue
        print(f"\nStrategy: {strat}")
        for fish in fish_types:
            fish_path = os.path.join(winners_path, fish)
            if not os.path.isdir(fish_path):
                continue
            best_file = None
            best_metrics = None
            best_score = (float('-inf'), float('-inf'), float('-inf'))
            for fname in os.listdir(fish_path):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(fish_path, fname), "r") as f:
                            data = json.load(f)
                        if not isinstance(data, dict):
                            continue  # Skip if not a dict
                        score = score_result(data)
                        if score > best_score:
                            best_score = score
                            best_file = fname
                            best_metrics = {
                                "sharpe_ratio": data.get("sharpe_ratio"),
                                "return_pct": data.get("return_pct"),
                                "profit_factor": data.get("profit_factor"),
                                "file": fname
                            }
                    except Exception as e:
                        print(f"    Error reading {fname}: {e}")
            if best_file and best_metrics is not None:
                print(f"  {fish}: {best_file}")
                print(f"    Sharpe: {best_metrics['sharpe_ratio']}, Return %: {best_metrics['return_pct']}, Profit Factor: {best_metrics['profit_factor']}")
            else:
                print(f"  {fish}: No JSON files found") 