import os
import pandas as pd
import numpy as np
import talib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'bull_segments')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
SMA_PERIOD = 100
ADX_PERIOD = 14
ADX_THRESHOLD = 25
RETURN_WINDOW = 30
RETURN_THRESHOLD = 0.05  # Try 5% for testing

# Bull segment extraction parameters
MIN_SEGMENT_LENGTH = 14  # Minimum data points required for backtesting
PADDING_BEFORE = 5      # Data points to include before bull segment
PADDING_AFTER = 5       # Data points to include after bull segment


def is_bull_market(df):
    df = df.copy()
    df['sma'] = df['Close'].rolling(window=SMA_PERIOD).mean()
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=ADX_PERIOD).mean()
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=ADX_PERIOD).sum() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=ADX_PERIOD).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx'] = dx.rolling(window=ADX_PERIOD).mean()
    df['return'] = df['Close'].pct_change(RETURN_WINDOW)
    # Bull market if price above SMA, ADX strong, and return positive
    df['bull'] = (
        (df['Close'] > df['sma']) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['return'] > RETURN_THRESHOLD)
    ).astype(int)
    return df

def extract_bull_segments(df, min_length=14, padding_before=5, padding_after=5):
    """
    Extract bull segments with padding to ensure minimum data length.
    
    Args:
        df: DataFrame with bull market data
        min_length: Minimum number of data points required
        padding_before: Number of data points to include before bull segment
        padding_after: Number of data points to include after bull segment
    """
    df = df.copy()
    df['bull_shift'] = df['bull'].shift(1, fill_value=0)
    df['segment'] = (df['bull'] != df['bull_shift']).cumsum()
    bull_segments = []
    
    for seg_id, seg_df in df.groupby('segment'):
        if seg_df['bull'].iloc[0] == 1:  # This is a bull segment
            # Get the start and end indices of this segment in the original dataframe
            start_idx = seg_df.index[0]
            end_idx = seg_df.index[-1]
            
            # Calculate padded indices
            padded_start = max(0, df.index.get_loc(start_idx) - padding_before)
            padded_end = min(len(df) - 1, df.index.get_loc(end_idx) + padding_after)
            
            # Extract the padded segment
            padded_segment = df.iloc[padded_start:padded_end + 1]
            
            # Only keep if it meets minimum length requirement
            if len(padded_segment) >= min_length:
                bull_segments.append(padded_segment)
    
    return bull_segments

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
        # Try to find the date column
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'timestamp', 'datetime']:
                date_col = col
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        # Standardize column names
        col_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=col_map)
        if not all(x in df.columns for x in ['Open', 'High', 'Low', 'Close', 'Volume']):
            print(f"Skipping {filepath}: missing OHLCV columns")
            return
        df = is_bull_market(df)
        bull_segments = extract_bull_segments(df, min_length=MIN_SEGMENT_LENGTH, padding_before=PADDING_BEFORE, padding_after=PADDING_AFTER)
        print(f"Processing {filepath}: {len(bull_segments)} bull segments found (with padding)")
        if len(bull_segments) == 0:
            print(f"  - Max bull run length: {df['bull'].sum()}")
            print(f"  - Any bull periods? {df['bull'].any()}")
            print(f"  - Data length: {len(df)}")
        for i, seg in enumerate(bull_segments):
            # Prepare output DataFrame with required columns and order
            if seg.index.name is not None and seg.index.name.lower() in ['date', 'time', 'timestamp', 'datetime']:
                seg_out = seg.copy()
                seg_out.reset_index(inplace=True)
            else:
                seg_out = seg.copy()
            # Ensure all required columns exist
            seg_out['Price'] = seg_out['Close']
            seg_out['Adj Close'] = seg_out['Close']
            # Reorder columns
            output_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price', 'Adj Close']
            # If the date column is named differently, rename it to 'Date'
            date_col = [c for c in seg_out.columns if c.lower() in ['date', 'time', 'timestamp', 'datetime']]
            if date_col and date_col[0] != 'Date':
                seg_out.rename(columns={date_col[0]: 'Date'}, inplace=True)
            seg_out = seg_out[output_cols]
            outname = os.path.splitext(os.path.basename(filepath))[0] + f'_bull_segment_{i}.csv'
            outpath = os.path.join(OUTPUT_DIR, outname)
            seg_out.to_csv(outpath, index=False)
            print(f"Saved bull segment: {outpath} ({len(seg_out)} rows, includes padding)")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith('.csv'):
            process_file(os.path.join(DATA_DIR, fname))

if __name__ == '__main__':
    main() 