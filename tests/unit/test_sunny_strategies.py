#!/usr/bin/env python3
"""
Test Runner for Sunny Harris Strategies

This script demonstrates how to use the three Sunny Harris strategies:
1. Sunny Bands Mean Reversion
2. DMA Histogram Momentum  
3. Fibonacci + Elliott Wave + Sunny Bands

It can be used with your existing test engines or run independently.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import strategies
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from newest_strats.sunny_bands_mean_reversion_strategy import SunnyBandsMeanReversionStrategy
from newest_strats.dma_histogram_momentum_strategy import DMAHistogramMomentumStrategy
from newest_strats.fibonacci_elliott_sunny_bands_strategy import FibonacciElliottSunnyBandsStrategy

def generate_sample_data(symbol: str = "SPY", days: int = 252, volatility: float = 0.02) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with trend and volatility
    initial_price = 100.0
    returns = np.random.normal(0.0005, volatility, len(dates))  # Slight upward bias
    
    # Add some trend
    trend = np.linspace(0, 0.1, len(dates))  # 10% upward trend over period
    returns += trend / len(dates)
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_vol = price * volatility / np.sqrt(252)
        
        # Random high/low range
        high_low_range = np.random.uniform(0.5, 2.0) * daily_vol
        
        high = price + high_low_range / 2
        low = price - high_low_range / 2
        
        # Ensure high >= low
        if high < low:
            high, low = low, high
        
        # Volume (random)
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df

def test_strategy(strategy_class, strategy_name: str, data: pd.DataFrame, parameters: dict = None):
    """Test a single strategy and display results"""
    print(f"\n{'='*60}")
    print(f"Testing {strategy_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize strategy
        strategy = strategy_class(parameters)
        
        # Run backtest
        start_time = datetime.now()
        results = strategy.backtest(data)
        end_time = datetime.now()
        
        # Display results
        print(f"Backtest completed in: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Final Cash: ${results['final_cash']:,.2f}")
        print(f"Number of Trades: {results['trades_count']}")
        
        if 'max_drawdown' in results:
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        if 'sharpe_ratio' in results:
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        
        if 'win_rate' in results:
            print(f"Win Rate: {results['win_rate']:.2%}")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        
        # Show sample trades
        if 'trades' in results and results['trades']:
            print(f"\nSample Trades (first 5):")
            for i, trade in enumerate(results['trades'][:5]):
                print(f"  {i+1}. {trade['type']} @ ${trade['price']:.2f}")
                if 'profit' in trade:
                    print(f"     Profit: ${trade['profit']:.2f}")
                elif 'loss' in trade:
                    print(f"     Loss: ${trade['loss']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"Error testing {strategy_name}: {str(e)}")
        return None

def run_all_strategies(data: pd.DataFrame):
    """Run all three Sunny Harris strategies"""
    print(f"Testing Sunny Harris Strategies")
    print(f"Data: {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    # Strategy 1: Sunny Bands Mean Reversion
    strategy1_params = {
        'dma_period': 20,
        'atr_period': 14,
        'band_multiplier': 2.0,
        'position_size': 0.1,
        'stop_loss_atr': 2.0
    }
    
    results1 = test_strategy(
        SunnyBandsMeanReversionStrategy, 
        "Sunny Bands Mean Reversion", 
        data, 
        strategy1_params
    )
    
    # Strategy 2: DMA Histogram Momentum
    strategy2_params = {
        'dma_period': 20,
        'atr_period': 14,
        'band_multiplier': 2.0,
        'position_size': 0.1,
        'stop_loss_atr': 1.5,
        'histogram_threshold': 0.001
    }
    
    results2 = test_strategy(
        DMAHistogramMomentumStrategy, 
        "DMA Histogram Momentum", 
        data, 
        strategy2_params
    )
    
    # Strategy 3: Fibonacci + Elliott Wave + Sunny Bands
    strategy3_params = {
        'dma_period': 20,
        'atr_period': 14,
        'band_multiplier': 2.0,
        'position_size': 0.1,
        'stop_loss_atr': 2.0,
        'elliott_wave_threshold': 0.05
    }
    
    results3 = test_strategy(
        FibonacciElliottSunnyBandsStrategy, 
        "Fibonacci + Elliott Wave + Sunny Bands", 
        data, 
        strategy3_params
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"STRATEGY COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    strategies = [
        ("Sunny Bands Mean Reversion", results1),
        ("DMA Histogram Momentum", results2),
        ("Fibonacci + Elliott + Sunny Bands", results3)
    ]
    
    for name, results in strategies:
        if results and 'error' not in results:
            print(f"{name:35} | Return: {results['total_return']:7.2%} | Trades: {results['trades_count']:3d} | Sharpe: {results.get('sharpe_ratio', 0):6.3f}")
        else:
            print(f"{name:35} | Error or no results")

def main():
    """Main function to run the test"""
    print("Sunny Harris Strategy Test Suite")
    print("=" * 60)
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(symbol="SPY", days=252, volatility=0.02)
    print(f"Generated {len(data)} days of sample data")
    
    # Run all strategies
    run_all_strategies(data)
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print("To use with your engines, import the strategy classes and call their methods.")
    print("Example:")
    print("  from newest_strats.sunny_bands_mean_reversion_strategy import SunnyBandsMeanReversionStrategy")
    print("  strategy = SunnyBandsMeanReversionStrategy()")
    print("  results = strategy.backtest(your_data)")

if __name__ == "__main__":
    main()
