#!/usr/bin/env python3
"""
Improved Pipeline Engine - Based on old_engines pipeline_engine patterns
Orchestrates multiple engines in sequence with comprehensive result analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import signal
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import inspect

# Suppress warnings
warnings.filterwarnings("ignore", message="A contingent SL/TP order would execute in the same bar*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backtesting\._stats")

@dataclass
class PipelineEngineConfig:
    """Configuration for pipeline engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Pipeline configuration
    engines_to_run: List[str] = None  # Will be set to all engines if None
    run_parallel: bool = False  # Run engines in parallel (not implemented yet)
    combine_results: bool = True
    save_combined_results: bool = True
    
    # Engine-specific configurations
    core_engine_config: Dict[str, Any] = None
    risk_engine_config: Dict[str, Any] = None
    statistical_engine_config: Dict[str, Any] = None
    
    # Performance optimization
    skip_existing_results: bool = True
    parallel_workers: int = 4
    
    # Results directory handling
    results_subdir_prefix: str = "improved_pipeline_backtest"
    
    # Output options
    save_json: bool = True
    save_csv: bool = True
    log_level: str = "INFO"

class PipelineEngine:
    """Pipeline engine that orchestrates multiple engines"""
    
    def __init__(self, config: PipelineEngineConfig = None):
        self.config = config or PipelineEngineConfig()
        self.setup_logging()
        self.setup_signal_handlers()
        self.results = {}
        self.interrupted = False
        
        # Set default engines to run
        if self.config.engines_to_run is None:
            self.config.engines_to_run = ['core', 'risk', 'statistical']
        
        # Initialize available engines
        self.available_engines = {
            'core': self.create_core_engine,
            'risk': self.create_risk_engine,
            'statistical': self.create_statistical_engine
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'improved_pipeline_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info("🛑 Interrupt signal received. Gracefully shutting down...")
            self.interrupted = True
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def create_core_engine(self):
        """Create core engine instance"""
        try:
            from .core_engine import CoreEngine, EngineConfig
            
            config = EngineConfig()
            if self.config.core_engine_config:
                for key, value in self.config.core_engine_config.items():
                    setattr(config, key, value)
            
            return CoreEngine(config)
        except ImportError:
            self.logger.error("❌ Could not import CoreEngine")
            return None
    
    def create_risk_engine(self):
        """Create risk engine instance"""
        try:
            from .risk_engine import RiskEngine, RiskEngineConfig
            
            config = RiskEngineConfig()
            if self.config.risk_engine_config:
                for key, value in self.config.risk_engine_config.items():
                    setattr(config, key, value)
            
            return RiskEngine(config)
        except ImportError:
            self.logger.error("❌ Could not import RiskEngine")
            return None
    
    def create_statistical_engine(self):
        """Create statistical engine instance"""
        try:
            from .statistical_engine import StatisticalEngine, StatisticalEngineConfig
            
            config = StatisticalEngineConfig()
            if self.config.statistical_engine_config:
                for key, value in self.config.statistical_engine_config.items():
                    setattr(config, key, value)
            
            return StatisticalEngine(config)
        except ImportError:
            self.logger.error("❌ Could not import StatisticalEngine")
            return None
    
    def run_engine(self, engine_name: str) -> Optional[Dict[str, Any]]:
        """Run a single engine"""
        if engine_name not in self.available_engines:
            self.logger.error(f"❌ Unknown engine: {engine_name}")
            return None
        
        try:
            self.logger.info(f"🚀 Starting {engine_name} engine...")
            start_time = time.time()
            
            # Create engine instance
            engine = self.available_engines[engine_name]()
            if engine is None:
                return None
            
            # Run engine
            engine.run()
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ {engine_name} engine completed in {execution_time:.2f}s")
            
            return {
                'engine_name': engine_name,
                'execution_time': execution_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error running {engine_name} engine: {e}")
            return {
                'engine_name': engine_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_sequential_pipeline(self) -> Dict[str, Any]:
        """Run engines in sequence"""
        self.logger.info("🔄 Running engines in sequence...")
        
        pipeline_results = {
            'pipeline_start_time': datetime.now().isoformat(),
            'engines_run': [],
            'total_execution_time': 0,
            'successful_engines': 0,
            'failed_engines': 0
        }
        
        start_time = time.time()
        
        for engine_name in self.config.engines_to_run:
            if self.interrupted:
                break
            
            self.logger.info(f"📊 Running {engine_name} engine...")
            engine_result = self.run_engine(engine_name)
            
            if engine_result:
                pipeline_results['engines_run'].append(engine_result)
                
                if engine_result['status'] == 'completed':
                    pipeline_results['successful_engines'] += 1
                else:
                    pipeline_results['failed_engines'] += 1
        
        pipeline_results['total_execution_time'] = time.time() - start_time
        pipeline_results['pipeline_end_time'] = datetime.now().isoformat()
        
        return pipeline_results
    
    def run_parallel_pipeline(self) -> Dict[str, Any]:
        """Run engines in parallel (not implemented yet)"""
        self.logger.warning("⚠️ Parallel pipeline execution not implemented yet, falling back to sequential")
        return self.run_sequential_pipeline()
    
    def combine_results(self) -> Dict[str, Any]:
        """Combine results from all engines"""
        if not self.config.combine_results:
            return {}
        
        self.logger.info("🔗 Combining results from all engines...")
        
        # Find all result files
        results_path = Path(self.config.results_path)
        if not results_path.exists():
            self.logger.warning("❌ Results path does not exist")
            return {}
        
        # Get all result directories
        result_dirs = [d for d in results_path.iterdir() if d.is_dir() and 'backtest' in d.name]
        if not result_dirs:
            self.logger.warning("❌ No result directories found")
            return {}
        
        # Sort by modification time (most recent first)
        result_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        combined_results = {
            'combination_timestamp': datetime.now().isoformat(),
            'engines_analyzed': [],
            'total_combinations': 0,
            'best_performers': {},
            'strategy_consistency': {},
            'data_file_analysis': {}
        }
        
        # Analyze each engine's results
        for result_dir in result_dirs[:5]:  # Analyze last 5 result directories
            try:
                # Look for JSON results
                json_files = list(result_dir.glob("*.json"))
                if not json_files:
                    continue
                
                # Load results
                with open(json_files[0], 'r') as f:
                    results = json.load(f)
                
                if not results:
                    continue
                
                df = pd.DataFrame(results)
                engine_name = result_dir.name.split('_')[0]
                
                # Analyze results
                engine_analysis = {
                    'engine_name': engine_name,
                    'total_combinations': len(results),
                    'strategies_tested': df['strategy_name'].nunique() if 'strategy_name' in df.columns else 0,
                    'data_files_tested': df['data_file'].nunique() if 'data_file' in df.columns else 0,
                    'best_return': df['total_return'].max() if 'total_return' in df.columns else 0,
                    'best_sharpe': df['sharpe_ratio'].max() if 'sharpe_ratio' in df.columns else 0,
                    'avg_quality_score': df['quality_score'].mean() if 'quality_score' in df.columns else 0,
                    'timestamp': result_dir.name
                }
                
                combined_results['engines_analyzed'].append(engine_analysis)
                combined_results['total_combinations'] += len(results)
                
                # Find best performers
                if 'total_return' in df.columns and 'strategy_name' in df.columns:
                    best_strategy = df.loc[df['total_return'].idxmax()]
                    combined_results['best_performers'][engine_name] = {
                        'strategy': best_strategy['strategy_name'],
                        'return': best_strategy['total_return'],
                        'sharpe': best_strategy.get('sharpe_ratio', 0),
                        'data_file': best_strategy.get('data_file', 'unknown')
                    }
                
            except Exception as e:
                self.logger.error(f"Error analyzing results from {result_dir.name}: {e}")
        
        # Calculate strategy consistency across engines
        if len(combined_results['engines_analyzed']) > 1:
            combined_results['strategy_consistency'] = self.calculate_strategy_consistency(combined_results)
        
        return combined_results
    
    def calculate_strategy_consistency(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate strategy consistency across engines"""
        consistency = {
            'consistent_strategies': [],
            'inconsistent_strategies': [],
            'consistency_score': 0.0
        }
        
        # This is a simplified implementation
        # In practice, you'd compare strategy performance across engines
        
        return consistency
    
    def save_pipeline_results(self, pipeline_results: Dict[str, Any], combined_results: Dict[str, Any]):
        """Save pipeline results"""
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_path) / f"{self.config.results_subdir_prefix}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline results
        if self.config.save_json:
            pipeline_path = results_dir / "pipeline_results.json"
            with open(pipeline_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            self.logger.info(f"✅ Pipeline results saved to: {pipeline_path}")
        
        # Save combined results
        if combined_results and self.config.save_combined_results:
            combined_path = results_dir / "combined_results.json"
            with open(combined_path, 'w') as f:
                json.dump(combined_results, f, indent=2)
            self.logger.info(f"✅ Combined results saved to: {combined_path}")
        
        # Save summary
        self.save_pipeline_summary(pipeline_results, combined_results, results_dir)
    
    def save_pipeline_summary(self, pipeline_results: Dict[str, Any], 
                            combined_results: Dict[str, Any], results_dir: Path):
        """Save pipeline summary"""
        summary = {
            'pipeline_summary': {
                'total_execution_time': pipeline_results.get('total_execution_time', 0),
                'successful_engines': pipeline_results.get('successful_engines', 0),
                'failed_engines': pipeline_results.get('failed_engines', 0),
                'engines_run': [r['engine_name'] for r in pipeline_results.get('engines_run', [])],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if combined_results:
            summary['combined_analysis'] = {
                'total_combinations_analyzed': combined_results.get('total_combinations', 0),
                'engines_analyzed': len(combined_results.get('engines_analyzed', [])),
                'best_performers': combined_results.get('best_performers', {}),
                'strategy_consistency': combined_results.get('strategy_consistency', {})
            }
        
        summary_path = results_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Pipeline Summary: {pipeline_results.get('successful_engines', 0)} engines completed, "
                        f"{combined_results.get('total_combinations', 0)} total combinations analyzed")
    
    def run_targeted_pipeline(self, target_strategies: List[str] = None, 
                            target_data_files: List[str] = None) -> Dict[str, Any]:
        """Run pipeline on specific strategies and data files"""
        self.logger.info("🎯 Running targeted pipeline...")
        
        # This would require modifying each engine to accept target filters
        # For now, we'll run the full pipeline
        return self.run()
    
    def run(self):
        """Main execution method"""
        start_time = datetime.now()
        self.logger.info("🚀 Starting Improved Pipeline Engine...")
        
        # Run pipeline
        if self.config.run_parallel:
            pipeline_results = self.run_parallel_pipeline()
        else:
            pipeline_results = self.run_sequential_pipeline()
        
        # Combine results
        combined_results = self.combine_results()
        
        # Save results
        self.save_pipeline_results(pipeline_results, combined_results)
        
        execution_time = datetime.now() - start_time
        self.logger.info(f"✅ Pipeline complete! Total execution time: {execution_time}")

def main():
    """Main entry point"""
    config = PipelineEngineConfig()
    engine = PipelineEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
