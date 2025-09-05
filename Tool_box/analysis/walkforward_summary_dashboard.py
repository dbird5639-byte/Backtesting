import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# --- User: Set your walk-forward results directory here ---
RESULTS_DIR = 'Backtesting/results'  # Root; loader will pick latest walkforward run

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .walkforward_results_loader import pick_latest_walkforward_run, load_walkforward_results

def collect_results(results_root):
    run_dir = pick_latest_walkforward_run(Path(results_root))
    if run_dir is None:
        return pd.DataFrame(), []
    df = load_walkforward_results(run_dir)
    # Backward-compat: no PNG collection integrated yet in new engines
    return df, []

def rank_strategies(df, metric='total_return'):
    """Rank strategies by mean of the given metric."""
    summary = df.groupby(['strategy_name', 'data_name'])[metric].mean().reset_index()
    summary = summary.sort_values(by=metric, ascending=False)
    return summary

def display_summary(summary, top_n=10):
    print("\nTop Strategies by Mean Walk-Forward Return:")
    print(summary.head(top_n).to_string(index=False))

def show_pngs(png_paths, top_strategies, max_per_strategy=2):
    """Display PNGs for top strategies (requires matplotlib inline or plt.show())."""
    shown = 0
    for strat, data, suffix, path in png_paths:
        if (strat, data) in top_strategies:
            img = plt.imread(path)
            plt.figure(figsize=(8, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{strat} | {data} | {suffix}')
            plt.show()
            shown += 1
            if shown >= max_per_strategy * len(top_strategies):
                break

def main():
    logger.info(f"Aggregating walk-forward results under: {RESULTS_DIR}")
    df, png_paths = collect_results(RESULTS_DIR)
    if df.empty:
        logger.warning("No walk-forward results found.")
        return
    # Rank strategies
    summary = rank_strategies(df, metric='total_return')
    display_summary(summary, top_n=10)
    # Optionally, show PNGs for top 3 strategies
    top_strats = set(zip(summary['strategy_name'].head(3), summary['data_name'].head(3)))
    show_pngs(png_paths, top_strats, max_per_strategy=3)
    # Save summary to CSV
    summary_path = os.path.join(RESULTS_DIR, 'walkforward_summary_ranking.csv')
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary ranking saved to: {summary_path}")

if __name__ == '__main__':
    main() 