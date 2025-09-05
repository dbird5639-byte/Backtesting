import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# --- User: Set your walk-forward results directory here ---
RESULTS_DIR = 'Backtesting/results/walkforward_20250708_153635'  # <-- Update as needed
DATA_DIR = str((Path(__file__).parent.parent / 'Data').resolve())

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def classify_regime(df, window=50, atr_window=14):
    # Trend/range: Use moving average slope
    ma = df['Close'].rolling(window=window).mean()
    slope = ma.diff()
    trend = np.where(slope > 0, 'trend_up', np.where(slope < 0, 'trend_down', 'range'))
    # Volatility: Use ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=atr_window).mean()
    median_atr = atr.median()
    volatility = np.where(atr > median_atr, 'high_vol', 'low_vol')
    return trend, volatility

def analyze_regimes(results_dir, data_dir):
    summary_records = []
    for strat_dir in Path(results_dir).iterdir():
        if not strat_dir.is_dir():
            continue
        for csv_file in strat_dir.glob('*.csv'):
            data_name = csv_file.stem.replace('_walkforward', '')
            # Try to find the corresponding data file
            data_path = Path(data_dir) / f'{data_name}.csv'
            if not data_path.exists():
                logger.warning(f"Data file not found for {data_name}")
                continue
            try:
                df_data = pd.read_csv(data_path)
                df_results = pd.read_csv(csv_file)
                # For each test window, classify regime by test_end date
                for _, row in df_results.iterrows():
                    test_end = pd.to_datetime(row['test_end'])
                    # Find the window in the data
                    window_df = df_data[df_data['Close'].notna()].copy()
                    window_df['Date'] = pd.to_datetime(window_df[window_df.columns[0]])
                    window_df = window_df[window_df['Date'] <= test_end].tail(50)  # last 50 bars up to test_end
                    if len(window_df) < 20:
                        continue
                    trend, volatility = classify_regime(window_df)
                    # Use the last value for this window
                    regime = f"{trend[-1]}_{volatility[-1]}"
                    summary_records.append({
                        'strategy': strat_dir.name,
                        'data': data_name,
                        'test_end': row['test_end'],
                        'regime': regime,
                        'total_return': row['total_return'],
                        'sharpe_ratio': row['sharpe_ratio'] if 'sharpe_ratio' in row else np.nan,
                        'max_drawdown': row['max_drawdown'] if 'max_drawdown' in row else np.nan
                    })
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
    if not summary_records:
        logger.warning("No regime analysis results found.")
        return
    df_summary = pd.DataFrame(summary_records)
    # Aggregate by regime
    agg = df_summary.groupby(['strategy', 'regime']).agg(
        mean_return=('total_return', 'mean'),
        mean_sharpe=('sharpe_ratio', 'mean'),
        mean_drawdown=('max_drawdown', 'mean'),
        count=('total_return', 'count')
    ).reset_index()
    print("\nRegime Analysis Summary:")
    print(agg.to_string(index=False))
    out_path = Path(results_dir) / 'regime_analysis_summary.csv'
    agg.to_csv(out_path, index=False)
    logger.info(f"Regime analysis summary saved to: {out_path}")

def main():
    analyze_regimes(RESULTS_DIR, DATA_DIR)

if __name__ == '__main__':
    main() 