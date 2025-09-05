#!/usr/bin/env python3
"""
Fibonacci/Gann Engine Example

This script demonstrates how to use the modern Fibonacci/Gann analysis engine
with comprehensive examples and best practices.
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from fibonacci_gann_engine import FibonacciGannEngine, FibonacciGannConfig
from engine_factory import EngineFactory, EngineFactoryConfig, EngineType, ExecutionMode


def create_sample_data():
    """Create sample market data for demonstration"""
    # Generate sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create realistic price movement with trends and volatility
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # Add trend and random walk
        trend = 0.001 * np.sin(i * 0.1)  # Cyclical trend
        random_change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + trend + random_change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


async def example_basic_usage():
    """Basic usage example of the Fibonacci/Gann engine"""
    print("=" * 60)
    print("FIBONACCI/GANN ENGINE - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    print(f"Created sample data with {len(data)} rows")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    
    # Create configuration
    config = FibonacciGannConfig(
        data_path="./sample_data",
        results_path="./Results",
        generate_charts=True,
        save_charts=True,
        interactive_charts=True,
        swing_window=5,
        min_swing_distance=0.02,  # 2% minimum swing
        gann_analysis_enabled=True,
        time_analysis_enabled=True
    )
    
    # Create engine
    engine = FibonacciGannEngine(config)
    
    # Run analysis
    print("\nRunning Fibonacci/Gann analysis...")
    results = await engine.run_analysis(data)
    
    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Detected {len(results.get('swing_points', []))} swing points")
        
        # Display key results
        if 'fibonacci_analysis' in results:
            fib_analysis = results['fibonacci_analysis']
            print(f"\nFibonacci Analysis:")
            if 'swing_high' in fib_analysis:
                swing_high = fib_analysis['swing_high']
                print(f"  Swing High: ${swing_high['price']:.2f} at {swing_high['timestamp']}")
            if 'swing_low' in fib_analysis:
                swing_low = fib_analysis['swing_low']
                print(f"  Swing Low: ${swing_low['price']:.2f} at {swing_low['timestamp']}")
            
            if 'retracement_levels' in fib_analysis:
                print(f"  Retracement Levels: {len(fib_analysis['retracement_levels'])} levels")
                for level_name, level_price in list(fib_analysis['retracement_levels'].items())[:3]:
                    print(f"    {level_name}: ${level_price:.2f}")
        
        if 'gann_analysis' in results:
            gann_analysis = results['gann_analysis']
            print(f"\nGann Analysis:")
            if 'gann_fan' in gann_analysis:
                print(f"  Gann Fan Lines: {len(gann_analysis['gann_fan'])} lines")
            if 'time_price_relationships' in gann_analysis:
                relationships = gann_analysis['time_price_relationships']
                print(f"  Price Velocity: {relationships.get('price_velocity_mean', 0):.4f}")
                print(f"  Detected Cycles: {len(relationships.get('detected_cycles', []))}")
    else:
        print("Analysis failed or returned no results")
    
    return results


async def example_advanced_configuration():
    """Advanced configuration example"""
    print("\n" + "=" * 60)
    print("FIBONACCI/GANN ENGINE - ADVANCED CONFIGURATION")
    print("=" * 60)
    
    # Create advanced configuration
    config = FibonacciGannConfig(
        data_path="./Data",
        results_path="./Results",
        
        # Custom Fibonacci ratios
        fibonacci_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        extension_ratios=[1.272, 1.618, 2.0, 2.618, 3.0, 3.618, 4.236],
        
        # Advanced swing detection
        swing_window=7,  # Larger window for more significant swings
        min_swing_distance=0.03,  # 3% minimum swing
        
        # Gann analysis settings
        gann_analysis_enabled=True,
        gann_square_size=11,  # Larger Gann square
        gann_angles=[1, 2, 3, 4, 6, 8, 12, 16],  # More Gann angles
        
        # Time analysis
        time_analysis_enabled=True,
        time_zone_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0],
        
        # Speed resistance
        speed_resistance_enabled=True,
        speed_angles=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        
        # Visualization
        generate_charts=True,
        save_charts=True,
        chart_format="html",
        interactive_charts=True,
        
        # Analysis settings
        lookback_periods=200,
        confidence_threshold=0.8
    )
    
    # Create sample data
    data = create_sample_data()
    
    # Create engine with advanced config
    engine = FibonacciGannEngine(config)
    
    print(f"Configuration Summary:")
    print(f"  Fibonacci Ratios: {len(config.fibonacci_ratios)} ratios")
    print(f"  Extension Ratios: {len(config.extension_ratios)} ratios")
    print(f"  Gann Angles: {len(config.gann_angles)} angles")
    print(f"  Swing Window: {config.swing_window}")
    print(f"  Min Swing Distance: {config.min_swing_distance*100:.1f}%")
    
    # Run analysis
    print(f"\nRunning advanced analysis...")
    results = await engine.run_analysis(data)
    
    if results:
        print(f"Advanced analysis completed!")
        
        # Display advanced results
        swing_points = results.get('swing_points', [])
        confirmed_swings = [sp for sp in swing_points if sp.get('confirmed', False)]
        print(f"Total swing points: {len(swing_points)}")
        print(f"Confirmed swing points: {len(confirmed_swings)}")
        
        # Show swing point details
        if confirmed_swings:
            print(f"\nConfirmed Swing Points:")
            for i, swing in enumerate(confirmed_swings[:5]):  # Show first 5
                print(f"  {i+1}. {swing['type'].upper()}: ${swing['price']:.2f} "
                      f"(Strength: {swing['strength']:.3f}) at {swing['timestamp']}")
    
    return results


async def example_engine_factory_integration():
    """Example using the Fibonacci/Gann engine through the Engine Factory"""
    print("\n" + "=" * 60)
    print("FIBONACCI/GANN ENGINE - FACTORY INTEGRATION")
    print("=" * 60)
    
    # Create factory configuration
    factory_config = EngineFactoryConfig(
        data_path="./Data",
        results_path="./Results",
        execution_mode=ExecutionMode.SEQUENTIAL,
        default_engines=[EngineType.FIBONACCI_GANN, EngineType.PERFORMANCE],
        fibonacci_gann_config={
            'generate_charts': True,
            'save_charts': True,
            'swing_window': 5,
            'min_swing_distance': 0.02,
            'gann_analysis_enabled': True,
            'time_analysis_enabled': True
        }
    )
    
    # Create factory
    factory = EngineFactory(factory_config)
    
    print(f"Factory Configuration:")
    print(f"  Execution Mode: {factory_config.execution_mode.value}")
    print(f"  Default Engines: {[e.value for e in factory_config.default_engines]}")
    print(f"  Fibonacci/Gann Config: {len(factory_config.fibonacci_gann_config)} parameters")
    
    # Get engine info
    fib_gann_info = factory.get_engine_info(EngineType.FIBONACCI_GANN)
    print(f"\nFibonacci/Gann Engine Info:")
    print(f"  Name: {fib_gann_info['name']}")
    print(f"  Description: {fib_gann_info['description']}")
    print(f"  Config Parameters: {len(fib_gann_info['config_parameters'])}")
    
    # Create sample data and save it
    data = create_sample_data()
    data_path = Path("./Data")
    data_path.mkdir(exist_ok=True)
    data.to_csv(data_path / "sample_market_data.csv", index=False)
    print(f"\nSaved sample data to {data_path / 'sample_market_data.csv'}")
    
    # Run factory
    print(f"\nRunning Engine Factory...")
    summary = await factory.run_factory()
    
    print(f"\nFactory Results:")
    print(f"  Total Engines: {summary.total_engines}")
    print(f"  Successful: {summary.successful_engines}")
    print(f"  Failed: {summary.failed_engines}")
    print(f"  Total Time: {summary.total_execution_time:.2f}s")
    
    for result in summary.engine_results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.engine_name}: {result.execution_time:.2f}s "
              f"({result.results_count} results)")
        if result.error_message:
            print(f"    Error: {result.error_message}")
    
    return summary


async def example_custom_analysis():
    """Example of custom analysis workflow"""
    print("\n" + "=" * 60)
    print("FIBONACCI/GANN ENGINE - CUSTOM ANALYSIS WORKFLOW")
    print("=" * 60)
    
    # Create sample data with specific patterns
    data = create_sample_data()
    
    # Add some artificial swing points for demonstration
    data.loc[50, 'high'] = data['close'].iloc[50] * 1.05  # Artificial swing high
    data.loc[100, 'low'] = data['close'].iloc[100] * 0.95  # Artificial swing low
    
    # Create custom configuration
    config = FibonacciGannConfig(
        data_path="./Data",
        results_path="./Results",
        swing_window=3,  # Smaller window for more sensitive detection
        min_swing_distance=0.01,  # 1% minimum swing
        fibonacci_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
        gann_angles=[1, 2, 3, 4, 6, 8],
        generate_charts=True,
        save_charts=True
    )
    
    # Create engine
    engine = FibonacciGannEngine(config)
    
    # Run step-by-step analysis
    print("Step 1: Detecting swing points...")
    await engine._detect_swing_points(data)
    print(f"Detected {len(engine.swing_points)} swing points")
    
    print("Step 2: Running Fibonacci analysis...")
    fib_results = await engine._run_fibonacci_analysis(data)
    print(f"Fibonacci analysis completed: {len(fib_results)} components")
    
    print("Step 3: Running Gann analysis...")
    gann_results = await engine._run_gann_analysis(data)
    print(f"Gann analysis completed: {len(gann_results)} components")
    
    print("Step 4: Generating visualizations...")
    if config.generate_charts:
        await engine._generate_visualizations(data, {
            'swing_points': [engine._swing_point_to_dict(sp) for sp in engine.swing_points],
            'fibonacci_analysis': fib_results,
            'gann_analysis': gann_results
        })
        print("Visualizations generated successfully")
    
    # Display detailed results
    print(f"\nDetailed Results:")
    print(f"  Swing Points: {len(engine.swing_points)}")
    for i, swing in enumerate(engine.swing_points[:3]):
        print(f"    {i+1}. {swing.type.upper()}: ${swing.price:.2f} "
              f"(Strength: {swing.strength:.3f})")
    
    if fib_results:
        print(f"  Fibonacci Retracements: {len(fib_results.get('retracement_levels', {}))}")
        print(f"  Fibonacci Extensions: {len(fib_results.get('extension_levels', {}))}")
    
    if gann_results:
        print(f"  Gann Fan Lines: {len(gann_results.get('gann_fan', {}))}")
        print(f"  Gann Box Levels: {len(gann_results.get('gann_box', {}))}")
    
    return {
        'swing_points': engine.swing_points,
        'fibonacci_results': fib_results,
        'gann_results': gann_results
    }


async def main():
    """Main function to run all examples"""
    print("FIBONACCI/GANN ENGINE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the modern Fibonacci/Gann analysis engine")
    print("with various configuration options and use cases.\n")
    
    try:
        # Run all examples
        await example_basic_usage()
        await example_advanced_configuration()
        await example_engine_factory_integration()
        await example_custom_analysis()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the ./Results directory for generated charts and analysis files.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    asyncio.run(main())
