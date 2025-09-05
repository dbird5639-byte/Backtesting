import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Path to organized results
RESULTS_PATH = r'C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results\bull_segments\basic_backtest_20250717_193928'

class BullRunAnalyzer:
    def __init__(self, results_path):
        self.results_path = results_path
        self.all_results = []
        self.strategy_summaries = {}
        
    def load_all_results(self):
        """Load all JSON results from the organized directory structure"""
        print("Loading all backtest results...")
        
        for strategy_dir in os.listdir(self.results_path):
            strategy_path = os.path.join(self.results_path, strategy_dir)
            if not os.path.isdir(strategy_path):
                continue
                
            strategy_results = []
            
            # Check each timeframe directory
            for timeframe in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                timeframe_path = os.path.join(strategy_path, timeframe)
                if not os.path.exists(timeframe_path):
                    continue
                    
                # Check each symbol directory
                for symbol_dir in os.listdir(timeframe_path):
                    symbol_path = os.path.join(timeframe_path, symbol_dir)
                    if not os.path.isdir(symbol_path):
                        continue
                        
                    # Load all JSON files in symbol directory
                    for json_file in os.listdir(symbol_path):
                        if json_file.endswith('.json'):
                            file_path = os.path.join(symbol_path, json_file)
                            try:
                                with open(file_path, 'r') as f:
                                    result = json.load(f)
                                    result['strategy'] = strategy_dir
                                    result['timeframe'] = timeframe
                                    result['symbol'] = symbol_dir
                                    result['segment_file'] = json_file
                                    strategy_results.append(result)
                            except Exception as e:
                                print(f"Error loading {file_path}: {e}")
            
            self.all_results.extend(strategy_results)
            print(f"Loaded {len(strategy_results)} results for {strategy_dir}")
        
        print(f"Total results loaded: {len(self.all_results)}")
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics for each strategy"""
        print("\nCalculating performance metrics...")
        
        # Group by strategy
        strategies = {}
        for result in self.all_results:
            strategy = result['strategy']
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(result)
        
        # Calculate metrics for each strategy
        for strategy, results in strategies.items():
            print(f"Analyzing {strategy}...")
            
            # Extract key metrics
            total_return = []
            sharpe_ratio = []
            max_drawdown = []
            win_rate = []
            profit_factor = []
            total_trades = []
            avg_trade_return = []
            success_count = 0
            
            for result in results:
                try:
                    # Basic metrics
                    if 'return_pct' in result:
                        total_return.append(result['return_pct'])
                    elif 'total_return' in result:
                        total_return.append(result['total_return'])
                    if 'sharpe_ratio' in result:
                        sharpe_ratio.append(result['sharpe_ratio'])
                    if 'max_drawdown' in result:
                        max_drawdown.append(result['max_drawdown'])
                    
                    # Trade metrics
                    if 'win_rate' in result:
                        win_rate.append(result['win_rate'])
                    if 'total_trades' in result:
                        total_trades.append(result['total_trades'])
                    if 'profit_factor' in result:
                        profit_factor.append(result['profit_factor'])
                    
                    # Calculate average trade return from total return and number of trades
                    if 'return_pct' in result and 'total_trades' in result and result['total_trades'] > 0:
                        avg_trade_return.append(result['return_pct'] / result['total_trades'])
                    elif 'total_return' in result and 'total_trades' in result and result['total_trades'] > 0:
                        avg_trade_return.append(result['total_return'] / result['total_trades'])
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing result: {e}")
                    continue
            
            # Calculate summary statistics
            if success_count > 0:
                self.strategy_summaries[strategy] = {
                    'total_tests': len(results),
                    'successful_tests': success_count,
                    'success_rate': success_count / len(results),
                    'avg_total_return': np.mean(total_return) if total_return else 0,
                    'median_total_return': np.median(total_return) if total_return else 0,
                    'avg_sharpe_ratio': np.mean(sharpe_ratio) if sharpe_ratio else 0,
                    'avg_max_drawdown': np.mean(max_drawdown) if max_drawdown else 0,
                    'avg_win_rate': np.mean(win_rate) if win_rate else 0,
                    'avg_profit_factor': np.mean(profit_factor) if profit_factor else 0,
                    'avg_total_trades': np.mean(total_trades) if total_trades else 0,
                    'avg_trade_return': np.mean(avg_trade_return) if avg_trade_return else 0,
                    'std_total_return': np.std(total_return) if total_return else 0,
                    'consistency_score': self.calculate_consistency_score(total_return),
                    'risk_adjusted_score': self.calculate_risk_adjusted_score(total_return, max_drawdown)
                }
    
    def calculate_consistency_score(self, returns):
        """Calculate consistency score based on positive returns percentage"""
        if not returns:
            return 0
        positive_returns = sum(1 for r in returns if r > 0)
        return positive_returns / len(returns)
    
    def calculate_risk_adjusted_score(self, returns, drawdowns):
        """Calculate risk-adjusted score"""
        if not returns or not drawdowns:
            return 0
        avg_return = np.mean(returns)
        avg_drawdown = np.mean(drawdowns)
        if avg_drawdown == 0:
            return avg_return
        return avg_return / abs(avg_drawdown)
    
    def generate_ranking_report(self):
        """Generate a comprehensive ranking report"""
        print("\n" + "="*80)
        print("BULL RUN STRATEGY PERFORMANCE ANALYSIS")
        print("="*80)
        
        if not self.strategy_summaries:
            print("No strategy summaries available. Run calculate_performance_metrics() first.")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame.from_dict(self.strategy_summaries, orient='index')
        
        # Add ranking columns
        df['return_rank'] = df['avg_total_return'].rank(ascending=False)
        df['sharpe_rank'] = df['avg_sharpe_ratio'].rank(ascending=False)
        df['consistency_rank'] = df['consistency_score'].rank(ascending=False)
        df['risk_adjusted_rank'] = df['risk_adjusted_score'].rank(ascending=False)
        
        # Calculate composite score (weighted average of ranks)
        df['composite_score'] = (
            df['return_rank'] * 0.3 +
            df['sharpe_rank'] * 0.25 +
            df['consistency_rank'] * 0.25 +
            df['risk_adjusted_rank'] * 0.2
        )
        df['composite_rank'] = df['composite_score'].rank()
        
        # Sort by composite rank
        df_sorted = df.sort_values('composite_rank')
        
        # Display top performers
        print("\n🏆 TOP 5 STRATEGIES FOR BULL RUNS:")
        print("-" * 80)
        
        for i, (strategy, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            strategy_name = str(strategy).upper() if strategy else "UNKNOWN"
            print(f"\n{i}. {strategy_name}")
            print(f"   📊 Composite Rank: {row['composite_rank']:.0f}")
            print(f"   💰 Avg Return: {row['avg_total_return']:.2%}")
            print(f"   📈 Sharpe Ratio: {row['avg_sharpe_ratio']:.2f}")
            print(f"   🎯 Win Rate: {row['avg_win_rate']:.2%}")
            print(f"   📉 Avg Max Drawdown: {row['avg_max_drawdown']:.2%}")
            print(f"   🔄 Consistency: {row['consistency_score']:.2%}")
            print(f"   ⚖️  Risk-Adjusted Score: {row['risk_adjusted_score']:.2f}")
            print(f"   📋 Success Rate: {row['success_rate']:.2%}")
        
        # Display detailed metrics for all strategies
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE METRICS")
        print("="*80)
        
        display_columns = [
            'composite_rank', 'avg_total_return', 'avg_sharpe_ratio', 
            'avg_win_rate', 'avg_max_drawdown', 'consistency_score',
            'avg_profit_factor', 'success_rate', 'avg_total_trades'
        ]
        
        print(df_sorted[display_columns].round(4))
        
        # Save detailed report
        report_path = os.path.join(self.results_path, 'bull_run_analysis_report.csv')
        df_sorted.to_csv(report_path)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return df_sorted
    
    def analyze_by_timeframe(self):
        """Analyze performance by timeframe"""
        print("\n" + "="*80)
        print("PERFORMANCE BY TIMEFRAME")
        print("="*80)
        
        timeframe_data = {}
        
        for result in self.all_results:
            timeframe = result['timeframe']
            if timeframe not in timeframe_data:
                timeframe_data[timeframe] = []
            timeframe_data[timeframe].append(result)
        
        for timeframe, results in timeframe_data.items():
            returns = []
            for r in results:
                if 'return_pct' in r:
                    returns.append(r['return_pct'])
                elif 'total_return' in r:
                    returns.append(r['total_return'])
            if returns:
                print(f"\n{timeframe.upper()}:")
                print(f"   📊 Avg Return: {np.mean(returns):.2%}")
                print(f"   📈 Median Return: {np.median(returns):.2%}")
                print(f"   📉 Std Dev: {np.std(returns):.2%}")
                print(f"   🎯 Positive Returns: {sum(1 for r in returns if r > 0)}/{len(returns)}")
    
    def analyze_by_symbol(self):
        """Analyze performance by symbol"""
        print("\n" + "="*80)
        print("TOP PERFORMING SYMBOLS")
        print("="*80)
        
        symbol_data = {}
        
        for result in self.all_results:
            symbol = result['symbol']
            if symbol not in symbol_data:
                symbol_data[symbol] = []
            symbol_data[symbol].append(result)
        
        symbol_performance = {}
        for symbol, results in symbol_data.items():
            returns = []
            for r in results:
                if 'return_pct' in r:
                    returns.append(r['return_pct'])
                elif 'total_return' in r:
                    returns.append(r['total_return'])
            if returns:
                symbol_performance[symbol] = {
                    'avg_return': np.mean(returns),
                    'count': len(returns),
                    'positive_rate': sum(1 for r in returns if r > 0) / len(returns)
                }
        
        # Sort by average return
        sorted_symbols = sorted(symbol_performance.items(), 
                              key=lambda x: x[1]['avg_return'], reverse=True)
        
        print("\nTop 10 Symbols by Average Return:")
        for i, (symbol, metrics) in enumerate(sorted_symbols[:10], 1):
            print(f"{i:2d}. {symbol:6s} | {metrics['avg_return']:6.2%} | "
                  f"{metrics['count']:3d} tests | {metrics['positive_rate']:5.1%} win rate")
    
    def generate_recommendations(self):
        """Generate specific recommendations for bot development"""
        print("\n" + "="*80)
        print("🤖 BOT DEVELOPMENT RECOMMENDATIONS")
        print("="*80)
        
        if not self.strategy_summaries:
            return
        
        df = pd.DataFrame.from_dict(self.strategy_summaries, orient='index')
        
        # High-performance strategies (good returns, low risk)
        high_performers = df[
            (df['avg_total_return'] > 0.05) &  # >5% return
            (df['avg_max_drawdown'] < 0.15) &  # <15% drawdown
            (df['consistency_score'] > 0.6)    # >60% consistency
        ]
        
        # Consistent performers (high win rate, good consistency)
        consistent_performers = df[
            (df['avg_win_rate'] > 0.6) &       # >60% win rate
            (df['consistency_score'] > 0.7) &  # >70% consistency
            (df['avg_total_return'] > 0.02)    # >2% return
        ]
        
        # High-frequency strategies (many trades, good profit factor)
        high_freq_performers = df[
            (df['avg_total_trades'] > 50) &    # >50 trades
            (df['avg_profit_factor'] > 1.2) &  # >1.2 profit factor
            (df['avg_total_return'] > 0.03)    # >3% return
        ]
        
        print("\n🎯 HIGH-PERFORMANCE STRATEGIES (Recommended for Primary Bots):")
        if not high_performers.empty:
            for strategy in high_performers.index:
                row = high_performers.loc[strategy]
                print(f"   • {strategy}: {row['avg_total_return']:.2%} return, "
                      f"{row['avg_max_drawdown']:.2%} drawdown, "
                      f"{row['consistency_score']:.1%} consistency")
        else:
            print("   No strategies meet high-performance criteria")
        
        print("\n🔄 CONSISTENT PERFORMERS (Good for Conservative Bots):")
        if not consistent_performers.empty:
            for strategy in consistent_performers.index:
                row = consistent_performers.loc[strategy]
                print(f"   • {strategy}: {row['avg_win_rate']:.1%} win rate, "
                      f"{row['consistency_score']:.1%} consistency, "
                      f"{row['avg_total_return']:.2%} return")
        else:
            print("   No strategies meet consistency criteria")
        
        print("\n⚡ HIGH-FREQUENCY STRATEGIES (Good for Scalping Bots):")
        if not high_freq_performers.empty:
            for strategy in high_freq_performers.index:
                row = high_freq_performers.loc[strategy]
                print(f"   • {strategy}: {row['avg_total_trades']:.0f} avg trades, "
                      f"{row['avg_profit_factor']:.2f} profit factor, "
                      f"{row['avg_total_return']:.2%} return")
        else:
            print("   No strategies meet high-frequency criteria")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Focus on the top 3-5 strategies from the ranking")
        print("   2. Test these strategies on out-of-sample bull market data")
        print("   3. Implement risk management and position sizing")
        print("   4. Start with paper trading before live deployment")
        print("   5. Monitor performance and adjust parameters as needed")

def main():
    """Run the complete bull run analysis"""
    analyzer = BullRunAnalyzer(RESULTS_PATH)
    
    # Load all results
    analyzer.load_all_results()
    
    # Calculate performance metrics
    analyzer.calculate_performance_metrics()
    
    # Generate comprehensive analysis
    analyzer.generate_ranking_report()
    analyzer.analyze_by_timeframe()
    analyzer.analyze_by_symbol()
    analyzer.generate_recommendations()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main() 