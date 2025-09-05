import os
import pandas as pd
import numpy as np

# Paths
BULL_SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'bull_segments')

def validate_file(filepath: str) -> dict:
    """Validate a single bull segment file"""
    issues = []
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns and df[col].isna().any():
                issues.append(f"NaN values in {col}: {df[col].isna().sum()} rows")
        
        # Check for zero or negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                zero_neg = (df[col] <= 0).sum()
                if zero_neg > 0:
                    issues.append(f"Zero/negative values in {col}: {zero_neg} rows")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = (
                (df['High'] < df['Low']) | 
                (df['Open'] > df['High']) | 
                (df['Open'] < df['Low']) |
                (df['Close'] > df['High']) | 
                (df['Close'] < df['Low'])
            ).sum()
            if invalid_ohlc > 0:
                issues.append(f"Invalid OHLC relationships: {invalid_ohlc} rows")
        
        # Check data length
        if len(df) < 5:
            issues.append(f"Too few data points: {len(df)} < 5")
        
        # Check for duplicate timestamps
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            duplicates = df['Date'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate timestamps: {duplicates} rows")
        
        return {
            'filename': os.path.basename(filepath),
            'rows': len(df),
            'issues': issues,
            'has_issues': len(issues) > 0
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'rows': 0,
            'issues': [f"Error reading file: {str(e)}"],
            'has_issues': True
        }

def main():
    """Validate all bull segment files"""
    print("Validating bull segment files...")
    print("=" * 50)
    
    all_files = []
    total_files = 0
    files_with_issues = 0
    
    for filename in os.listdir(BULL_SEGMENTS_DIR):
        if filename.endswith('.csv'):
            total_files += 1
            filepath = os.path.join(BULL_SEGMENTS_DIR, filename)
            result = validate_file(filepath)
            all_files.append(result)
            
            if result['has_issues']:
                files_with_issues += 1
                print(f"\n❌ {result['filename']} ({result['rows']} rows)")
                for issue in result['issues']:
                    print(f"   - {issue}")
            else:
                print(f"✅ {result['filename']} ({result['rows']} rows)")
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"Total files: {total_files}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Clean files: {total_files - files_with_issues}")
    
    if files_with_issues > 0:
        print(f"\nRecommendation: Fix issues in {files_with_issues} files before running backtests")
    else:
        print(f"\nAll files look good! Ready for backtesting.")

if __name__ == '__main__':
    main() 