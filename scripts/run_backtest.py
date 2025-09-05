#!/usr/bin/env python3
"""
Simple Backtest Runner

This script demonstrates how to use the new master backtesting architecture.
It runs a simple moving average strategy backtest and shows the results.
"""

import sys
import os
from pathlib import Path

# Add the core package to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.base import EngineConfig, StrategyConfig
from core.engines import SimpleEngine


def main():
    """Main function to run a simple backtest"""
    print("🚀 Master Backtesting System - Simple Demo")
    print("=" * 50)
    
    # Create engine configuration
    config = EngineConfig(
        data_path="./data",
        strategies_path="./strategies",
        results_path="./results",
        initial_cash=100000.0,
        commission=0.002,
        slippage=0.0001,
        save_results=True,
        save_plots=False,
        verbose=True
    )
    
    print(f"📊 Engine Configuration:")
    print(f"   Initial Cash: ${config.initial_cash:,.2f}")
    print(f"   Commission: {config.commission:.3f}")
    print(f"   Slippage: {config.slippage:.4f}")
    print()
    
    # Initialize the simple engine
    print("🔧 Initializing Simple Backtesting Engine...")
    engine = SimpleEngine(config)
    
    # Load market data
    print("📈 Loading market data...")
    data = engine.load_data("BTC", "1h")
    print(f"   Loaded {len(data)} data points")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    print()
    
    # Load strategy
    print("🎯 Loading trading strategy...")
    strategy = engine.load_strategy(
        "SimpleMAStrategy",
        symbol="BTC",
        timeframe="1h",
        short_window=10,
        long_window=20
    )
    print(f"   Strategy: {strategy.config.name}")
    print(f"   Symbol: {strategy.config.symbol}")
    print(f"   Timeframe: {strategy.config.timeframe}")
    print()
    
    # Run backtest
    print("🏃 Running backtest...")
    result = engine.run_backtest(strategy, data)
    
    # Display results
    print("📊 Backtest Results:")
    print("=" * 50)
    print(f"Strategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Timeframe: {result.timeframe}")
    print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print()
    
    print("Performance Metrics:")
    print(f"   Total Return: {result.total_return:.2%}")
    print(f"   Annualized Return: {result.annualized_return:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print()
    
    print("Trade Statistics:")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Winning Trades: {result.winning_trades}")
    print(f"   Losing Trades: {result.losing_trades}")
    print()
    
    print("Risk Metrics:")
    print(f"   Volatility: {result.volatility:.2%}")
    print(f"   VaR (95%): {result.var_95:.2%}")
    print(f"   CVaR (95%): {result.cvar_95:.2%}")
    print()
    
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print()
    
    # Save results
    print("💾 Saving results...")
    engine.save_results(result)
    print(f"   Results saved to: {config.results_path}")
    
    # Show summary
    print("\n🎉 Backtest completed successfully!")
    print("=" * 50)
    
    # Get summary stats
    summary = engine.get_summary_stats()
    if summary:
        print("Summary Statistics:")
        print(f"   Total Backtests: {summary['total_backtests']}")
        print(f"   Average Return: {summary['avg_return']:.2%}")
        print(f"   Average Sharpe: {summary['avg_sharpe']:.2f}")
        print(f"   Best Strategy: {summary['best_strategy']}")
    
    print("\n✨ This demonstrates the new Master Backtesting Architecture!")
    print("   - Clean, organized structure")
    print("   - Consistent interfaces")
    print("   - Easy to extend and customize")
    print("   - Professional-grade backtesting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
