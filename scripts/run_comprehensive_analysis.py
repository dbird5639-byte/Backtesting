#!/usr/bin/env python3
"""
Comprehensive Analysis Runner Script

This script demonstrates how to use the Master Orchestrator Engine to run
comprehensive analysis across all engines, data files, and strategies.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engines.master_orchestrator_engine import (
    MasterOrchestratorEngine, 
    MasterOrchestratorConfig,
    SimpleMAStrategy
)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('comprehensive_analysis.log')
        ]
    )

def create_sample_strategies() -> List[SimpleMAStrategy]:
    """Create sample strategies for testing"""
    strategies = []
    
    # Strategy 1: Short-term momentum
    strategy1 = SimpleMAStrategy(
        name="Short_Momentum",
        parameters={'short_window': 5, 'long_window': 15}
    )
    strategies.append(strategy1)
    
    # Strategy 2: Medium-term trend following
    strategy2 = SimpleMAStrategy(
        name="Medium_Trend",
        parameters={'short_window': 20, 'long_window': 50}
    )
    strategies.append(strategy2)
    
    # Strategy 3: Long-term trend following
    strategy3 = SimpleMAStrategy(
        name="Long_Trend",
        parameters={'short_window': 50, 'long_window': 200}
    )
    strategies.append(strategy3)
    
    return strategies

def find_data_files(data_directory: str = "./data") -> List[str]:
    """Find data files in the specified directory"""
    data_files = []
    
    if not os.path.exists(data_directory):
        print(f"Data directory {data_directory} does not exist. Creating sample data...")
        create_sample_data(data_directory)
    
    # Look for common data file formats
    for file in os.listdir(data_directory):
        if file.endswith(('.csv', '.parquet', '.json')):
            data_files.append(os.path.join(data_directory, file))
    
    if not data_files:
        print("No data files found. Creating sample data...")
        create_sample_data(data_directory)
        data_files = find_data_files(data_directory)
    
    return data_files

def create_sample_data(data_directory: str):
    """Create sample data files for testing"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    os.makedirs(data_directory, exist_ok=True)
    
    # Create sample price data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 150, len(dates))
    
    # Add some cyclical patterns
    cycle1 = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Annual cycle
    cycle2 = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 63)    # Quarterly cycle
    
    # Add random noise
    noise = np.random.normal(0, 2, len(dates))
    
    # Combine all components
    close_prices = trend + cycle1 + cycle2 + noise
    
    # Generate OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.normal(0, 0.5, len(dates)),
        'high': close_prices + np.abs(np.random.normal(0, 1, len(dates))),
        'low': close_prices - np.abs(np.random.normal(0, 1, len(dates))),
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.5, len(dates))
    })
    
    # Ensure high >= close >= low
    data['high'] = data[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 0.5, len(dates)))
    data['low'] = data[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 0.5, len(dates)))
    
    # Save as CSV
    csv_file = os.path.join(data_directory, "sample_price_data.csv")
    data.to_csv(csv_file, index=False)
    print(f"Created sample data: {csv_file}")
    
    # Create a second sample with different characteristics
    dates2 = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    # Different trend pattern
    trend2 = 200 - np.linspace(0, 50, len(dates2))
    cycle2_1 = 15 * np.sin(2 * np.pi * np.arange(len(dates2)) / 126)  # Semi-annual cycle
    
    close_prices2 = trend2 + cycle2_1 + np.random.normal(0, 3, len(dates2))
    
    data2 = pd.DataFrame({
        'timestamp': dates2,
        'open': close_prices2 + np.random.normal(0, 0.8, len(dates2)),
        'high': close_prices2 + np.abs(np.random.normal(0, 1.5, len(dates2))),
        'low': close_prices2 - np.abs(np.random.normal(0, 1.5, len(dates2))),
        'close': close_prices2,
        'volume': np.random.lognormal(9, 0.7, len(dates2))
    })
    
    # Ensure high >= close >= low
    data2['high'] = data2[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 0.8, len(dates2)))
    data2['low'] = data2[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 0.8, len(dates2)))
    
    # Save as CSV
    csv_file2 = os.path.join(data_directory, "sample_price_data_2.csv")
    data2.to_csv(csv_file2, index=False)
    print(f"Created sample data: {csv_file2}")

def run_comprehensive_analysis():
    """Run comprehensive analysis using the master orchestrator engine"""
    try:
        print("Starting Comprehensive Analysis...")
        print("=" * 50)
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Create configuration
        config = MasterOrchestratorConfig(
            # Engine configurations
            simple_engine_config=None,  # Use defaults
            advanced_engine_config=None,  # Use defaults
            permutation_engine_config=None,  # Use defaults
            risk_engine_config=None,  # Use defaults
            walkforward_engine_config=None,  # Use defaults
            fibonacci_engine_config=None,  # Use defaults
            regime_analysis_config=None,  # Use defaults
            
            # Orchestration settings
            parallel_processing=True,
            max_workers=2,  # Reduced for demonstration
            engine_timeout=600,  # 10 minutes
            
            # Data processing
            data_directory="./data",
            results_directory="./results",
            cache_results=True,
            
            # Analysis settings
            generate_comprehensive_analysis=True,
            generate_cross_engine_comparison=True,
            generate_regime_based_analysis=True,
            generate_performance_heatmaps=True,
            
            # Output settings
            save_detailed_results=True,
            generate_interactive_charts=True,
            export_to_excel=False
        )
        
        # Initialize master orchestrator engine
        print("Initializing Master Orchestrator Engine...")
        orchestrator = MasterOrchestratorEngine(config)
        
        # Find data files
        print("Finding data files...")
        data_files = find_data_files(config.data_directory)
        print(f"Found {len(data_files)} data files:")
        for file in data_files:
            print(f"  - {file}")
        
        # Create strategies
        print("Creating strategies...")
        strategies = create_sample_strategies()
        print(f"Created {len(strategies)} strategies:")
        for strategy in strategies:
            print(f"  - {strategy.name}: {strategy.parameters}")
        
        # Run comprehensive analysis
        print("\nRunning comprehensive analysis...")
        print("This may take several minutes...")
        
        comprehensive_result = orchestrator.run_comprehensive_analysis(data_files, strategies)
        
        print("Comprehensive analysis completed successfully!")
        
        # Save results
        print("Saving results...")
        output_path = orchestrator.save_comprehensive_results(comprehensive_result)
        print(f"Results saved to: {output_path}")
        
        # Display summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        
        summary = comprehensive_result.summary_statistics
        print(f"Total analyses: {summary.get('total_analyses', 0)}")
        print(f"Successful analyses: {summary.get('successful_analyses', 0)}")
        print(f"Failed analyses: {summary.get('failed_analyses', 0)}")
        
        # Display engine rankings
        best_engines = summary.get('best_performing_engines', {})
        if 'rankings' in best_engines:
            print(f"\nEngine Performance Rankings:")
            for i, engine in enumerate(best_engines['rankings'], 1):
                score = best_engines.get('scores', {}).get(engine, 0)
                print(f"  {i}. {engine}: {score:.4f}")
        
        # Display regime summary
        regime_summary = summary.get('regime_summary', {})
        print(f"\nRegime Analysis Summary:")
        print(f"  Total regimes detected: {regime_summary.get('total_regimes_detected', 0)}")
        print(f"  Average regime duration: {regime_summary.get('average_regime_duration', 0):.1f} days")
        print(f"  Total regime transitions: {regime_summary.get('regime_transitions', 0)}")
        
        # Display data coverage
        data_coverage = summary.get('data_coverage', {})
        print(f"\nData Coverage:")
        print(f"  Total data files: {data_coverage.get('total_data_files', 0)}")
        print(f"  Common files across engines: {len(data_coverage.get('common_files', []))}")
        
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print(f"Check the results directory for detailed outputs: {output_path}")
        
        return comprehensive_result
        
    except Exception as e:
        print(f"Error running comprehensive analysis: {e}")
        logger.error(f"Error running comprehensive analysis: {e}", exc_info=True)
        raise

def main():
    """Main function"""
    try:
        # Run comprehensive analysis
        result = run_comprehensive_analysis()
        
        # Additional analysis can be performed here
        print("\nComprehensive analysis completed!")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
