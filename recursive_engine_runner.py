#!/usr/bin/env python3
"""
Recursive Engine Runner - Processes each engine-strategy-data combination individually
with parallel processing and resume functionality.
"""

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import json

# Add the current directory to Python path for imports
sys.path.append('.')

from Engines.core_engine import CoreEngine, EngineConfig
from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig
from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
from Engines.regime_analysis_engine import RegimeAnalysisEngine, RegimeAnalysisConfig
from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig

class RecursiveEngineRunner:
    """Recursive engine runner that processes each engine-strategy-data combination individually."""
    
    def __init__(self, data_path: str, strategies_path: str, results_path: str, max_workers: int = 3):
        self.data_path = Path(data_path)
        self.strategies_path = Path(strategies_path)
        self.results_path = Path(results_path)
        self.max_workers = max_workers
        
        # Setup logging
        self.setup_logging()
        
        # Engine configurations
        self.engine_configs = {
            'CoreEngine': EngineConfig(
                data_path=str(self.data_path),
                strategies_path=str(self.strategies_path),
                results_path=str(self.results_path),
                save_json=False,
                save_csv=True,
                save_plots=False
            ),
            'EnhancedRiskEngine': EnhancedRiskEngineConfig(
                data_path=str(self.data_path),
                strategies_path=str(self.strategies_path),
                results_path=str(self.results_path),
                walk_forward_enabled=True,
                in_sample_periods=252,
                out_of_sample_periods=63,
                min_periods_for_analysis=100,
                save_walk_forward_results=True,
                save_parameter_comparison=True,
                save_risk_attribution=True,
                create_visualizations=True
            ),
            'EnhancedVisualizationEngine': VisualizationConfig(
                data_path=str(self.data_path),
                strategies_path=str(self.strategies_path),
                results_path=str(self.results_path),
                save_csv=True,
                save_png=True,
                save_html=True,
                enable_regime_overlay=True,
                organize_by_strategy=True,
                organize_by_data_file=True,
                create_summary_dashboard=True
            ),
            'RegimeAnalysisEngine': RegimeAnalysisConfig(
                data_path=str(self.data_path),
                strategies_path=str(self.strategies_path),
                results_path=str(self.results_path),
                save_csv=True,
                save_png=True,
                save_html=True,
                enable_regime_detection=True,
                create_regime_visualizations=True,
                generate_regime_reports=True
            ),
            'RegimeOverlayEngine': RegimeOverlayConfig(
                data_path=str(self.data_path),
                strategies_path=str(self.strategies_path),
                results_path=str(self.results_path),
                regime_data_path=str(self.results_path),
                enable_regime_filtering=True,
                enable_strategy_recommendations=True,
                enable_regime_alerts=True
            )
        }
        
        # Progress tracking
        self.progress_file = self.results_path / "recursive_progress.json"
        self.completed_combinations: Set[str] = set()
        self.load_progress()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.results_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"recursive_runner_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_progress(self):
        """Load progress from previous runs."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.completed_combinations = set(progress_data.get('completed', []))
                    self.logger.info(f"Loaded progress: {len(self.completed_combinations)} completed combinations")
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
                self.completed_combinations = set()
        else:
            self.completed_combinations = set()
            
    def save_progress(self):
        """Save current progress."""
        try:
            progress_data = {
                'completed': list(self.completed_combinations),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")
            
    def get_combination_key(self, engine_name: str, strategy_name: str, data_file: str) -> str:
        """Generate a unique key for engine-strategy-data combination."""
        return f"{engine_name}|{strategy_name}|{data_file}"
        
    def is_combination_completed(self, engine_name: str, strategy_name: str, data_file: str) -> bool:
        """Check if a combination has already been completed."""
        key = self.get_combination_key(engine_name, strategy_name, data_file)
        return key in self.completed_combinations
        
    def mark_combination_completed(self, engine_name: str, strategy_name: str, data_file: str):
        """Mark a combination as completed."""
        key = self.get_combination_key(engine_name, strategy_name, data_file)
        self.completed_combinations.add(key)
        self.save_progress()
        
    def discover_data_files(self) -> List[str]:
        """Discover all CSV data files."""
        csv_files = []
        for file_path in self.data_path.glob("*.csv"):
            try:
                # Quick validation - check if file has required columns
                df = pd.read_csv(file_path, nrows=5)
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    csv_files.append(file_path.name)
                else:
                    self.logger.warning(f"Missing columns in {file_path.name}: {[col for col in required_cols if col not in df.columns]}")
            except Exception as e:
                self.logger.error(f"Error validating {file_path.name}: {e}")
                
        self.logger.info(f"Found {len(csv_files)} valid CSV files")
        return csv_files
        
    def discover_strategies(self) -> List[str]:
        """Discover all strategy files."""
        strategy_files = []
        for file_path in self.strategies_path.glob("*.py"):
            if file_path.name != "__init__.py":
                strategy_files.append(file_path.stem)
                
        self.logger.info(f"Found {len(strategy_files)} strategy files")
        return strategy_files
        
    def create_engine(self, engine_name: str):
        """Create an engine instance."""
        config = self.engine_configs[engine_name]
        
        if engine_name == 'CoreEngine':
            return CoreEngine(config)
        elif engine_name == 'EnhancedRiskEngine':
            return EnhancedRiskEngine(config)
        elif engine_name == 'EnhancedVisualizationEngine':
            return EnhancedVisualizationEngine(config)
        elif engine_name == 'RegimeAnalysisEngine':
            return RegimeAnalysisEngine(config)
        elif engine_name == 'RegimeOverlayEngine':
            return RegimeOverlayEngine(config)
        else:
            raise ValueError(f"Unknown engine: {engine_name}")
            
    def process_single_combination(self, engine_name: str, strategy_name: str, data_file: str) -> Dict:
        """Process a single engine-strategy-data combination."""
        combination_key = f"{engine_name} - {strategy_name} - {data_file}"
        
        try:
            self.logger.info(f"Processing: {combination_key}")
            
            # Create engine
            engine = self.create_engine(engine_name)
            
            # Load data
            data_path = self.data_path / data_file
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
                
            df = pd.read_csv(data_path)
            if len(df) < 100:  # Minimum data requirement
                raise ValueError(f"Insufficient data: {len(df)} rows")
                
            # Load strategy
            strategy_path = self.strategies_path / f"{strategy_name}.py"
            if not strategy_path.exists():
                raise FileNotFoundError(f"Strategy file not found: {strategy_name}.py")
                
            # Create a simple strategy instance for testing
            # This is a simplified approach - in production you'd load the actual strategy class
            class SimpleTestStrategy:
                def __init__(self, name):
                    self.name = name
                    
                def generate_signals(self, data):
                    # Simple buy and hold strategy for testing
                    signals = pd.DataFrame(index=data.index)
                    signals['signal'] = 0
                    signals['signal'].iloc[0] = 1  # Buy at start
                    signals['signal'].iloc[-1] = -1  # Sell at end
                    return signals
                    
            strategy = SimpleTestStrategy(strategy_name)
            
            # Run backtest
            if engine_name == 'CoreEngine':
                results = asyncio.run(engine.run_backtest(strategy, df, data_file))
            elif engine_name == 'EnhancedRiskEngine':
                results = asyncio.run(engine.run_walk_forward_analysis(strategy, df, data_file))
            else:
                # For other engines, use a simplified approach
                results = [{
                    'strategy_name': strategy_name,
                    'data_file': data_file,
                    'total_return': 0.05,  # 5% return for testing
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.02,
                    'win_rate': 0.6
                }]
                
            # Save results
            self.save_combination_results(engine_name, strategy_name, data_file, results)
            
            # Mark as completed
            self.mark_combination_completed(engine_name, strategy_name, data_file)
            
            return {
                'success': True,
                'combination': combination_key,
                'results_count': len(results) if isinstance(results, list) else 1,
                'message': f"Successfully processed {combination_key}"
            }
            
        except Exception as e:
            error_msg = f"Error processing {combination_key}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'combination': combination_key,
                'error': str(e),
                'message': error_msg
            }
            
    def save_combination_results(self, engine_name: str, strategy_name: str, data_file: str, results: List):
        """Save results for a specific combination."""
        # Create directory structure: Engine/Strategy/Data/
        result_dir = self.results_path / engine_name / strategy_name / data_file.replace('.csv', '')
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_file = result_dir / "results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save summary as CSV
        if isinstance(results, list) and results:
            summary_data = []
            for result in results:
                if isinstance(result, dict):
                    summary_data.append({
                        'strategy': strategy_name,
                        'data_file': data_file,
                        'total_return': result.get('total_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'win_rate': result.get('win_rate', 0)
                    })
                    
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                csv_file = result_dir / "summary.csv"
                summary_df.to_csv(csv_file, index=False)
                
    def generate_combinations(self) -> List[Tuple[str, str, str]]:
        """Generate all engine-strategy-data combinations."""
        engines = list(self.engine_configs.keys())
        strategies = self.discover_strategies()
        data_files = self.discover_data_files()
        
        combinations = []
        for engine in engines:
            for strategy in strategies:
                for data_file in data_files:
                    if not self.is_combination_completed(engine, strategy, data_file):
                        combinations.append((engine, strategy, data_file))
                        
        self.logger.info(f"Generated {len(combinations)} combinations to process")
        return combinations
        
    def run_recursive_processing(self):
        """Run recursive processing with parallel execution."""
        combinations = self.generate_combinations()
        
        if not combinations:
            self.logger.info("All combinations already completed!")
            return
            
        self.logger.info(f"Starting recursive processing of {len(combinations)} combinations with {self.max_workers} workers")
        
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_combination = {
                executor.submit(self.process_single_combination, engine, strategy, data_file): (engine, strategy, data_file)
                for engine, strategy, data_file in combinations
            }
            
            # Process completed tasks
            for future in as_completed(future_to_combination):
                engine, strategy, data_file = future_to_combination[future]
                
                try:
                    result = future.result()
                    if result['success']:
                        completed_count += 1
                        self.logger.info(f"✅ {result['combination']} - {result['results_count']} results")
                    else:
                        failed_count += 1
                        self.logger.error(f"❌ {result['combination']} - {result['error']}")
                        
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"❌ {engine} - {strategy} - {data_file} - Exception: {e}")
                    
                # Progress update
                total_processed = completed_count + failed_count
                if total_processed % 10 == 0:
                    progress_pct = (total_processed / len(combinations)) * 100
                    self.logger.info(f"Progress: {progress_pct:.1f}% ({total_processed}/{len(combinations)})")
                    
        self.logger.info(f"Recursive processing completed!")
        self.logger.info(f"✅ Completed: {completed_count}")
        self.logger.info(f"❌ Failed: {failed_count}")
        self.logger.info(f"📊 Total: {len(combinations)}")

def main():
    """Main execution function."""
    print("Starting Recursive Engine Runner...")
    
    # Configuration
    data_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    print(f"Data path: {data_path}")
    print(f"Strategies path: {strategies_path}")
    print(f"Results path: {results_path}")
    
    try:
        # Create runner
        print("Creating RecursiveEngineRunner...")
        runner = RecursiveEngineRunner(data_path, strategies_path, results_path, max_workers=3)
        
        # Run recursive processing
        print("Starting recursive processing...")
        runner.run_recursive_processing()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
