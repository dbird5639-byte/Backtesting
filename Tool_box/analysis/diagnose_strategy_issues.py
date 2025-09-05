import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Path to organized results
RESULTS_PATH = r'C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results\bull_segments\basic_backtest_20250717_193928'

def analyze_strategy_issues():
    """Analyze why strategies are performing poorly during bull runs"""
    print("🔍 DIAGNOSING STRATEGY PERFORMANCE ISSUES")
    print("=" * 80)
    
    # Load sample results for each strategy
    strategy_samples = {}
    
    for strategy_dir in os.listdir(RESULTS_PATH):
        strategy_path = os.path.join(RESULTS_PATH, strategy_dir)
        if not os.path.isdir(strategy_path):
            continue
            
        # Get a few sample results
        samples = []
        sample_count = 0
        
        for timeframe in ['1h', '4h', '1d']:
            if sample_count >= 5:  # Limit samples per strategy
                break
            timeframe_path = os.path.join(strategy_path, timeframe)
            if not os.path.exists(timeframe_path):
                continue
                
            for symbol_dir in os.listdir(timeframe_path):
                if sample_count >= 5:
                    break
                symbol_path = os.path.join(timeframe_path, symbol_dir)
                if not os.path.isdir(symbol_path):
                    continue
                    
                for json_file in os.listdir(symbol_path):
                    if sample_count >= 5:
                        break
                    if json_file.endswith('.json'):
                        file_path = os.path.join(symbol_path, json_file)
                        try:
                            with open(file_path, 'r') as f:
                                result = json.load(f)
                                result['strategy'] = strategy_dir
                                result['timeframe'] = timeframe
                                result['symbol'] = symbol_dir
                                result['file'] = json_file
                                samples.append(result)
                                sample_count += 1
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        
        strategy_samples[strategy_dir] = samples
    
    # Analyze each strategy
    for strategy, samples in strategy_samples.items():
        print(f"\n📊 {strategy.upper()}")
        print("-" * 60)
        
        if not samples:
            print("   No samples found")
            continue
        
        # Key metrics analysis
        returns = [s.get('return_pct', 0) for s in samples]
        trades = [s.get('total_trades', 0) for s in samples]
        win_rates = [s.get('win_rate', 0) for s in samples]
        drawdowns = [s.get('max_drawdown', 0) for s in samples]
        exposures = [s.get('exposure_time', 0) for s in samples]
        
        print(f"   📈 Sample Size: {len(samples)}")
        print(f"   💰 Avg Return: {np.mean(returns):.2%}")
        print(f"   📉 Avg Drawdown: {np.mean(drawdowns):.2%}")
        print(f"   🔄 Avg Trades: {np.mean(trades):.1f}")
        print(f"   🎯 Avg Win Rate: {np.mean(win_rates):.2%}")
        print(f"   ⏱️  Avg Exposure: {np.mean(exposures):.1f}%")
        
        # Identify potential issues
        issues = []
        
        if np.mean(returns) < -0.5:  # >50% loss
            issues.append("❌ Extremely poor returns")
        
        if np.mean(trades) < 2:
            issues.append("⚠️  Very few trades (may be overfitting)")
        
        if np.mean(win_rates) > 2:  # >200% win rate (impossible)
            issues.append("❌ Invalid win rate calculation")
        
        if np.mean(drawdowns) < -5:  # >500% drawdown
            issues.append("❌ Excessive drawdowns")
        
        if np.mean(exposures) < 10:  # <10% exposure
            issues.append("⚠️  Very low market exposure")
        
        if issues:
            print("   🚨 POTENTIAL ISSUES:")
            for issue in issues:
                print(f"      {issue}")
        else:
            print("   ✅ No obvious issues detected")
        
        # Show a detailed sample
        if samples:
            sample = samples[0]
            print(f"\n   📋 Sample Result ({sample['symbol']} {sample['timeframe']}):")
            print(f"      Return: {sample.get('return_pct', 0):.2%}")
            print(f"      Trades: {sample.get('total_trades', 0)}")
            print(f"      Win Rate: {sample.get('win_rate', 0):.2%}")
            print(f"      Drawdown: {sample.get('max_drawdown', 0):.2%}")
            print(f"      Exposure: {sample.get('exposure_time', 0):.1f}%")
            print(f"      Buy & Hold: {sample.get('buy_hold_return', 0):.2%}")

def analyze_timeframe_performance():
    """Analyze performance across different timeframes"""
    print("\n" + "=" * 80)
    print("⏰ TIMEFRAME PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    timeframe_data = defaultdict(list)
    
    for strategy_dir in os.listdir(RESULTS_PATH):
        strategy_path = os.path.join(RESULTS_PATH, strategy_dir)
        if not os.path.isdir(strategy_path):
            continue
            
        for timeframe in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
            timeframe_path = os.path.join(strategy_path, timeframe)
            if not os.path.exists(timeframe_path):
                continue
                
            for symbol_dir in os.listdir(timeframe_path):
                symbol_path = os.path.join(timeframe_path, symbol_dir)
                if not os.path.isdir(symbol_path):
                    continue
                    
                for json_file in os.listdir(symbol_path):
                    if json_file.endswith('.json'):
                        file_path = os.path.join(symbol_path, json_file)
                        try:
                            with open(file_path, 'r') as f:
                                result = json.load(f)
                                timeframe_data[timeframe].append(result)
                        except:
                            continue
    
    for timeframe, results in timeframe_data.items():
        if not results:
            continue
            
        returns = [r.get('return_pct', 0) for r in results]
        trades = [r.get('total_trades', 0) for r in results]
        win_rates = [r.get('win_rate', 0) for r in results]
        exposures = [r.get('exposure_time', 0) for r in results]
        
        print(f"\n{timeframe.upper()}:")
        print(f"   📊 Results: {len(results)}")
        print(f"   💰 Avg Return: {np.mean(returns):.2%}")
        print(f"   📈 Median Return: {np.median(returns):.2%}")
        print(f"   🔄 Avg Trades: {np.mean(trades):.1f}")
        print(f"   🎯 Avg Win Rate: {np.mean(win_rates):.2%}")
        print(f"   ⏱️  Avg Exposure: {np.mean(exposures):.1f}%")
        print(f"   📉 Positive Returns: {sum(1 for r in returns if r > 0)}/{len(returns)}")

def suggest_improvements():
    """Suggest improvements based on the analysis"""
    print("\n" + "=" * 80)
    print("💡 IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🚨 CRITICAL ISSUES IDENTIFIED:")
    print("   1. All strategies showing negative returns during bull runs")
    print("   2. Extremely high drawdowns (>500% in many cases)")
    print("   3. Invalid win rate calculations (>200%)")
    print("   4. Very low market exposure (<10% in many cases)")
    print("   5. Very few trades per test (potential overfitting)")
    
    print("\n🔧 RECOMMENDED FIXES:")
    print("\n   1. STRATEGY LOGIC REVIEW:")
    print("      • Check if strategies are correctly identifying bull market conditions")
    print("      • Verify entry/exit logic is appropriate for trending markets")
    print("      • Ensure position sizing is reasonable")
    
    print("\n   2. RISK MANAGEMENT:")
    print("      • Implement stop-loss mechanisms")
    print("      • Add maximum position size limits")
    print("      • Consider trailing stops for profit protection")
    
    print("\n   3. PARAMETER OPTIMIZATION:")
    print("      • Test different parameter ranges for each strategy")
    print("      • Use walk-forward analysis to avoid overfitting")
    print("      • Consider adaptive parameters based on market conditions")
    
    print("\n   4. DATA QUALITY:")
    print("      • Verify bull segment extraction is working correctly")
    print("      • Check for data gaps or anomalies")
    print("      • Ensure proper OHLC data validation")
    
    print("\n   5. BACKTESTING ENGINE:")
    print("      • Review commission and slippage settings")
    print("      • Check if position sizing is implemented correctly")
    print("      • Verify trade execution logic")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Start with the 30M timeframe (showing positive returns)")
    print("   2. Focus on MATIC and DOGE (best performing symbols)")
    print("   3. Review and fix strategy logic issues")
    print("   4. Implement proper risk management")
    print("   5. Re-run backtests with improvements")
    print("   6. Consider using a different backtesting framework if issues persist")

def main():
    """Run the complete diagnostic analysis"""
    analyze_strategy_issues()
    analyze_timeframe_performance()
    suggest_improvements()
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSIS COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main() 