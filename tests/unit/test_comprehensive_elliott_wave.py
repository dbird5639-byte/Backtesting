#!/usr/bin/env python3
"""
Test Comprehensive Elliott Wave Strategy

This script demonstrates how to use the comprehensive Elliott Wave strategy
with the specialized Elliott Wave engine.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Engines'))

from comprehensive_elliott_wave_strategy import ComprehensiveElliottWaveStrategy
from Engines.elliott_wave_engine import ElliottWaveEngine

def generate_sample_data(periods: int = 500) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    
    # Generate price data with Elliott Wave patterns
    base_price = 100.0
    prices = [base_price]
    
    # Create some Elliott Wave patterns
    for i in range(1, periods):
        # Add some trend and Elliott Wave characteristics
        if i < 100:  # Wave 1
            change = np.random.normal(0.5, 0.3)
        elif i < 150:  # Wave 2 (correction)
            change = np.random.normal(-0.3, 0.2)
        elif i < 250:  # Wave 3 (strongest)
            change = np.random.normal(0.8, 0.4)
        elif i < 300:  # Wave 4 (correction)
            change = np.random.normal(-0.2, 0.15)
        elif i < 400:  # Wave 5
            change = np.random.normal(0.4, 0.25)
        else:  # ABC correction
            if i < 450:
                change = np.random.normal(-0.4, 0.2)
            else:
                change = np.random.normal(0.2, 0.1)
        
        new_price = prices[-1] * (1 + change/100)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = price * 0.02  # 2% volatility
        
        high = price + np.random.uniform(0, volatility)
        low = price - np.random.uniform(0, volatility)
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC relationships
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

def test_comprehensive_elliott_wave_strategy():
    """Test the comprehensive Elliott Wave strategy"""
    print("Testing Comprehensive Elliott Wave Strategy")
    print("=" * 50)
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(500)
    print(f"Generated {len(data)} bars of data")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print()
    
    # Test strategy directly
    print("Testing strategy directly...")
    strategy = ComprehensiveElliottWaveStrategy()
    
    # Calculate signals
    signals = strategy.calculate_signals(data)
    signal_count = len(signals[signals != 0])
    print(f"Generated {signal_count} trading signals")
    
    # Find peaks and patterns
    peaks = strategy.find_peaks(data)
    print(f"Identified {len(peaks)} significant peaks")
    
    # Identify different wave patterns
    impulse_waves = strategy.identify_impulse_waves(peaks, data)
    correction_waves = strategy.identify_correction_waves(peaks, data)
    triangle_waves = strategy.identify_triangle_waves(peaks, data)
    double_combos = strategy.identify_double_combos(peaks, data)
    triple_combos = strategy.identify_triple_combos(peaks, data)
    
    print(f"Patterns identified:")
    print(f"  - Impulse waves: {len(impulse_waves)}")
    print(f"  - Correction waves: {len(correction_waves)}")
    print(f"  - Triangle waves: {len(triangle_waves)}")
    print(f"  - Double combos: {len(double_combos)}")
    print(f"  - Triple combos: {len(triple_combos)}")
    
    # Analyze time cycles
    time_cycles = strategy.analyze_time_cycles(peaks, data)
    print(f"Time cycles analyzed: {len(time_cycles['wave_durations'])} wave durations")
    print()
    
    # Test backtest method
    print("Testing strategy backtest method...")
    backtest_results = strategy.backtest(data)
    
    if 'error' in backtest_results:
        print(f"Backtest error: {backtest_results['error']}")
    else:
        print(f"Backtest completed successfully:")
        print(f"  - Total return: {backtest_results['total_return']:.2%}")
        print(f"  - Final cash: ${backtest_results['final_cash']:,.2f}")
        print(f"  - Trades count: {backtest_results['trades_count']}")
        print(f"  - Max drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"  - Sharpe ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"  - Win rate: {backtest_results['win_rate']:.2%}")
    print()
    
    return strategy, data

def test_elliott_wave_engine(strategy, data):
    """Test the Elliott Wave engine with the strategy"""
    print("Testing Elliott Wave Engine")
    print("=" * 50)
    
    # Create engine
    engine_params = {
        'initial_cash': 100000.0,
        'commission': 0.001,
        'max_positions': 3,
        'risk_per_trade': 0.02,
        'position_sizing': 'risk_based',
        'stop_loss_type': 'atr',
        'take_profit_type': 'fibonacci',
        'enable_pattern_recognition': True,
        'enable_time_cycles': True,
        'enable_fibonacci_validation': True
    }
    
    engine = ElliottWaveEngine(engine_params)
    print(f"Engine created with parameters: {engine_params}")
    
    # Load strategy
    print("Loading strategy into engine...")
    if engine.load_strategy(ComprehensiveElliottWaveStrategy):
        print("Strategy loaded successfully")
    else:
        print("Failed to load strategy")
        return
    
    # Run backtest
    print("Running backtest with engine...")
    results = engine.run_backtest(data)
    
    if 'error' in results:
        print(f"Engine backtest error: {results['error']}")
        return
    
    # Display results
    print("Engine backtest completed successfully!")
    print()
    
    # Backtest results
    backtest_results = results['backtest_results']
    print("Backtest Results:")
    print(f"  - Final cash: ${backtest_results['final_cash']:,.2f}")
    print(f"  - Total trades: {backtest_results['total_trades']}")
    print(f"  - Final positions: {backtest_results['final_positions']}")
    print()
    
    # Performance metrics
    performance = results['performance_metrics']
    print("Performance Metrics:")
    print(f"  - Total return: {performance['total_return']:.2%}")
    print(f"  - Max drawdown: {performance['max_drawdown']:.2%}")
    print(f"  - Volatility: {performance['volatility']:.2%}")
    print(f"  - Win rate: {performance['win_rate']:.2%}")
    print(f"  - Profit factor: {performance['profit_factor']:.2f}")
    print(f"  - Winning trades: {performance['winning_trades']}")
    print(f"  - Losing trades: {performance['losing_trades']}")
    print()
    
    # Pattern analysis
    patterns = results['patterns_identified']
    print(f"Pattern Analysis:")
    print(f"  - Total patterns identified: {len(patterns)}")
    
    if patterns:
        pattern_types = {}
        for pattern in patterns:
            ptype = pattern['pattern_type']
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        for ptype, count in pattern_types.items():
            print(f"    - {ptype}: {count}")
    
    # Wave counts
    wave_counts = results['wave_counts']
    print(f"  - Wave count updates: {len(wave_counts)}")
    
    # Time cycles
    time_cycles = results['time_cycles']
    print(f"  - Time cycle updates: {len(time_cycles)}")
    print()
    
    # Display engine summary
    print("Engine Summary:")
    print(engine.get_results_summary())

def main():
    """Main test function"""
    print("Comprehensive Elliott Wave Strategy Test")
    print("=" * 60)
    print()
    
    try:
        # Test strategy directly
        strategy, data = test_comprehensive_elliott_wave_strategy()
        
        print("-" * 60)
        print()
        
        # Test with engine
        test_elliott_wave_engine(strategy, data)
        
        print("-" * 60)
        print()
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
