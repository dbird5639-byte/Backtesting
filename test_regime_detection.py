#!/usr/bin/env python3
"""
Test script for regime detection system - tests a single data file
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append('.')

def test_regime_detection():
    """Test regime detection on a single data file."""
    print("Testing Regime Detection System...")
    
    try:
        # Import engines
        from Engines.regime_detection_engine import RegimeDetectionEngine, RegimeDetectionConfig
        from Engines.regime_visualization_engine import RegimeVisualizationEngine, RegimeVisualizationConfig
        
        # Configuration
        data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
        results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
        
        # Create configurations
        regime_config = RegimeDetectionConfig(
            data_path=data_path,
            results_path=results_path,
            n_regimes=3,  # Use fewer regimes for testing
            min_regime_size=20  # Lower minimum for testing
        )
        
        viz_config = RegimeVisualizationConfig(
            data_path=data_path,
            results_path=results_path
        )
        
        # Create engines
        regime_engine = RegimeDetectionEngine(regime_config)
        viz_engine = RegimeVisualizationEngine(viz_config)
        
        # Find a test data file
        data_dir = Path(data_path)
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in data directory")
            return
        
        # Use the first CSV file for testing
        test_file = csv_files[0]
        print(f"Testing with file: {test_file.name}")
        
        # Load data
        data = regime_engine.load_data(str(test_file))
        if data.empty:
            print(f"No valid data loaded from {test_file.name}")
            return
        
        print(f"Loaded data: {len(data)} rows")
        print(f"Data columns: {list(data.columns)}")
        
        # Detect regimes
        print("Detecting market regimes...")
        regime_result = regime_engine.detect_market_regimes(data, test_file.name)
        
        print(f"Detected {regime_result.total_regimes} regimes")
        print(f"Regime types: {list(set([regime.regime_type for regime in regime_result.regime_mapping.values()]))}")
        
        # Save regime results
        print("Saving regime results...")
        regime_output_path = regime_engine.save_regime_results(regime_result)
        print(f"Regime results saved to: {regime_output_path}")
        
        # Generate visualizations
        print("Generating visualizations...")
        viz_output_path = viz_engine.generate_regime_report(regime_result, test_file.name)
        print(f"Visualizations saved to: {viz_output_path}")
        
        print("✅ Regime detection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in regime detection test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_regime_detection()
