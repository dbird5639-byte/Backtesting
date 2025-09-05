import os
import pandas as pd
import shutil

# Paths
BULL_SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'bull_segments')
USABLE_SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'usable_bull_segments')
MIN_LENGTH = 14  # Minimum data points required

def filter_segments():
    """Filter bull segments by length and copy usable ones to a separate folder"""
    os.makedirs(USABLE_SEGMENTS_DIR, exist_ok=True)
    
    total_files = 0
    usable_files = 0
    
    for filename in os.listdir(BULL_SEGMENTS_DIR):
        if filename.endswith('.csv'):
            total_files += 1
            filepath = os.path.join(BULL_SEGMENTS_DIR, filename)
            
            try:
                df = pd.read_csv(filepath)
                if len(df) >= MIN_LENGTH:
                    # Copy to usable folder
                    dest_path = os.path.join(USABLE_SEGMENTS_DIR, filename)
                    shutil.copy2(filepath, dest_path)
                    usable_files += 1
                    print(f"✓ {filename}: {len(df)} rows")
                else:
                    print(f"✗ {filename}: {len(df)} rows (too short)")
            except Exception as e:
                print(f"✗ {filename}: Error reading file - {e}")
    
    print(f"\nSummary:")
    print(f"Total files: {total_files}")
    print(f"Usable files: {usable_files}")
    print(f"Filtered out: {total_files - usable_files}")
    print(f"Usable files copied to: {USABLE_SEGMENTS_DIR}")

if __name__ == '__main__':
    filter_segments() 