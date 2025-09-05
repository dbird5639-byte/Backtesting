#!/usr/bin/env python3
# Resume Backtesting Script
# Generated on 2025-08-05 17:50:47

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from run_engines import *

def resume_backtesting():
    """Resume backtesting from where it left off"""
    print("🔄 Resuming Backtesting...")
    print("=" * 40)
    

    # Resume basic_backtest_20250802_204712
    print("\n🚀 Resuming basic_backtest_20250802_204712...")
    try:
        run_basic_engine()
        print("✅ basic_backtest_20250802_204712 completed successfully")
    except Exception as e:
        print(f"❌ Error resuming basic_backtest_20250802_204712: {e}")

if __name__ == "__main__":
    resume_backtesting()
