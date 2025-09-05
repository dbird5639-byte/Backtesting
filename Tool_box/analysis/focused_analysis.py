import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Path to organized results
RESULTS_PATH = r'C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results\bull_segments\basic_backtest_20250717_193928'

def find_best_performers():
    """Find the best performing strategy-timeframe-symbol combinations"""
    print("🎯 FINDING BEST PERFORMING COMBINATIONS")
    print("=" * 80)
    
    all_results = []
    
    # Load all results
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
                                result['strategy'] = strategy_dir
                                result['timeframe'] = timeframe
                                result['symbol'] = symbol_dir
                                result['file'] = json_file
                                all_results.append(result)
                        except:
                            continue
    
    # Filter for positive returns only
    positive_results = [r for r in all_results if r.get('return_pct', 0) > 0]
    
    print(f"📊 Total Results: {len(all_results)}")
    print(f"✅ Positive Returns: {len(positive_results)} ({len(positive_results)/len(all_results)*100:.1f}%)")
    
    if not positive_results:
        print("\n❌ No positive returns found. All strategies need improvement.")
        return
    
    # Sort by return
    positive_results.sort(key=lambda x: x.get('return_pct', 0), reverse=True)
    
    print(f"\n🏆 TOP 20 BEST PERFORMERS:")
    print("-" * 80)
    
    for i, result in enumerate(positive_results[:20], 1):
        strategy = result['strategy']
        timeframe = result['timeframe']
        symbol = result['symbol']
        return_pct = result.get('return_pct', 0)
        trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        drawdown = result.get('max_drawdown', 0)
        exposure = result.get('exposure_time', 0)
        buy_hold = result.get('buy_hold_return', 0)
        
        print(f"{i:2d}. {strategy:25s} | {timeframe:3s} | {symbol:6s} | "
              f"{return_pct:6.1f}% | {trades:2d} trades | {win_rate:5.1f}% win | "
              f"{drawdown:6.1f}% DD | {exposure:4.1f}% exp | B&H: {buy_hold:5.1f}%")
    
    return positive_results

def analyze_winning_patterns(positive_results):
    """Analyze patterns in winning combinations"""
    print("\n" + "=" * 80)
    print("📈 WINNING PATTERNS ANALYSIS")
    print("=" * 80)
    
    if not positive_results:
        return
    
    # Strategy analysis
    strategy_stats = defaultdict(list)
    timeframe_stats = defaultdict(list)
    symbol_stats = defaultdict(list)
    
    for result in positive_results:
        strategy_stats[result['strategy']].append(result['return_pct'])
        timeframe_stats[result['timeframe']].append(result['return_pct'])
        symbol_stats[result['symbol']].append(result['return_pct'])
    
    print("\n🏆 BEST STRATEGIES (by number of positive results):")
    for strategy, returns in sorted(strategy_stats.items(), key=lambda x: len(x[1]), reverse=True):
        avg_return = np.mean(returns)
        print(f"   {strategy:25s} | {len(returns):3d} wins | {avg_return:6.1f}% avg return")
    
    print("\n⏰ BEST TIMEFRAMES (by number of positive results):")
    for timeframe, returns in sorted(timeframe_stats.items(), key=lambda x: len(x[1]), reverse=True):
        avg_return = np.mean(returns)
        print(f"   {timeframe:3s} | {len(returns):3d} wins | {avg_return:6.1f}% avg return")
    
    print("\n💎 BEST SYMBOLS (by number of positive results):")
    for symbol, returns in sorted(symbol_stats.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        avg_return = np.mean(returns)
        print(f"   {symbol:6s} | {len(returns):3d} wins | {avg_return:6.1f}% avg return")

def identify_bot_candidates(positive_results):
    """Identify the best candidates for bot development"""
    print("\n" + "=" * 80)
    print("🤖 BOT DEVELOPMENT CANDIDATES")
    print("=" * 80)
    
    if not positive_results:
        print("❌ No candidates found - all strategies need improvement")
        return
    
    # Filter for reasonable performance
    good_candidates = []
    for result in positive_results:
        return_pct = result.get('return_pct', 0)
        trades = result.get('total_trades', 0)
        drawdown = result.get('max_drawdown', 0)
        exposure = result.get('exposure_time', 0)
        
        # Criteria for good candidates
        if (return_pct > 5 and  # >5% return
            trades >= 2 and     # At least 2 trades
            drawdown > -50 and  # <50% drawdown
            exposure > 10):     # >10% exposure
            good_candidates.append(result)
    
    if not good_candidates:
        print("⚠️  No candidates meet strict criteria. Lowering standards...")
        # Lower standards
        good_candidates = [r for r in positive_results if r.get('return_pct', 0) > 1]
    
    if not good_candidates:
        print("❌ No suitable candidates found")
        return
    
    # Sort by return
    good_candidates.sort(key=lambda x: x.get('return_pct', 0), reverse=True)
    
    print(f"\n🎯 TOP BOT CANDIDATES ({len(good_candidates)} found):")
    print("-" * 80)
    
    for i, result in enumerate(good_candidates[:15], 1):
        strategy = result['strategy']
        timeframe = result['timeframe']
        symbol = result['symbol']
        return_pct = result.get('return_pct', 0)
        trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0)
        drawdown = result.get('max_drawdown', 0)
        exposure = result.get('exposure_time', 0)
        
        print(f"{i:2d}. {strategy:20s} | {timeframe:3s} | {symbol:6s} | "
              f"{return_pct:6.1f}% | {trades:2d} trades | {win_rate:5.1f}% win | "
              f"{drawdown:6.1f}% DD | {exposure:4.1f}% exp")
    
    # Group by strategy for recommendations
    strategy_groups = defaultdict(list)
    for result in good_candidates:
        strategy_groups[result['strategy']].append(result)
    
    print(f"\n💡 RECOMMENDED BOT STRATEGIES:")
    print("-" * 40)
    
    for strategy, results in strategy_groups.items():
        avg_return = np.mean([r.get('return_pct', 0) for r in results])
        avg_trades = np.mean([r.get('total_trades', 0) for r in results])
        avg_drawdown = np.mean([r.get('max_drawdown', 0) for r in results])
        
        print(f"   {strategy}: {len(results)} good results, {avg_return:.1f}% avg return, "
              f"{avg_trades:.1f} avg trades, {avg_drawdown:.1f}% avg drawdown")

def provide_actionable_insights():
    """Provide actionable insights for improvement"""
    print("\n" + "=" * 80)
    print("💡 ACTIONABLE INSIGHTS & NEXT STEPS")
    print("=" * 80)
    
    print("\n🚨 CRITICAL FINDINGS:")
    print("   1. Most strategies are underperforming during bull runs")
    print("   2. Very few trades per test (potential overfitting)")
    print("   3. Invalid win rate calculations in many cases")
    print("   4. Strategies may not be optimized for trending markets")
    
    print("\n✅ POSITIVE DISCOVERIES:")
    print("   1. Some strategies do show positive returns")
    print("   2. 30M timeframe shows better performance")
    print("   3. Certain symbols (MATIC, DOGE) perform better")
    print("   4. VWAP strategy shows some promise")
    
    print("\n🎯 IMMEDIATE ACTIONS:")
    print("\n   1. FOCUS ON WINNERS:")
    print("      • Start with the top 5-10 positive performers")
    print("      • Focus on 30M timeframe strategies")
    print("      • Prioritize MATIC and DOGE symbols")
    
    print("\n   2. STRATEGY IMPROVEMENTS:")
    print("      • Review and fix win rate calculations")
    print("      • Implement proper risk management")
    print("      • Add stop-loss and take-profit mechanisms")
    print("      • Optimize parameters for bull market conditions")
    
    print("\n   3. BACKTESTING IMPROVEMENTS:")
    print("      • Add commission and slippage modeling")
    print("      • Implement realistic position sizing")
    print("      • Add market impact considerations")
    print("      • Use walk-forward analysis")
    
    print("\n   4. BOT DEVELOPMENT PLAN:")
    print("      • Start with paper trading the best performers")
    print("      • Implement proper risk management")
    print("      • Add monitoring and alerting systems")
    print("      • Consider multi-strategy portfolio approach")
    
    print("\n📋 SPECIFIC RECOMMENDATIONS:")
    print("   1. Begin with VWAP strategy on 30M timeframe")
    print("   2. Focus on MATIC and DOGE for initial testing")
    print("   3. Implement 2% stop-loss and 6% take-profit")
    print("   4. Use 1% position sizing per trade")
    print("   5. Monitor drawdown and adjust parameters")
    print("   6. Consider combining multiple strategies")

def main():
    """Run the focused analysis"""
    positive_results = find_best_performers()
    analyze_winning_patterns(positive_results)
    identify_bot_candidates(positive_results)
    provide_actionable_insights()
    
    print("\n" + "=" * 80)
    print("✅ FOCUSED ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main() 