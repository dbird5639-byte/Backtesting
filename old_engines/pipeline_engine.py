"""
Pipeline Backtesting Engine

This engine orchestrates the execution of multiple backtesting engines
and combines their results for comprehensive strategy analysis.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from base_engine import BaseEngine, EngineConfig
from basic_engine import BasicEngine, BasicEngineConfig
from risk_managed_engine import RiskManagedEngine, RiskManagedEngineConfig
from statistical_engine import StatisticalEngine, StatisticalEngineConfig
from walkforward_engine import WalkforwardEngine, WalkforwardConfig
from alpha_engine import AlphaEngine, AlphaEngineConfig
from epoch_engine import EpochEngine, EpochEngineConfig
from ml_engine import MLEngine, MLEngineConfig

@dataclass
class PipelineEngineConfig(EngineConfig):
    """Configuration for pipeline backtesting engine"""
    # Pipeline configuration
    engines_to_run: List[str] = None  # Will be set to all engines if None
    
    # Engine-specific configurations
    basic_engine_config: BasicEngineConfig = None
    risk_managed_engine_config: RiskManagedEngineConfig = None
    statistical_engine_config: StatisticalEngineConfig = None
    walkforward_engine_config: WalkforwardConfig = None
    alpha_engine_config: AlphaEngineConfig = None
    # Risk-managed variants
    risk_managed_oos_engine_config: RiskManagedEngineConfig = None
    risk_managed_is_engine_config: RiskManagedEngineConfig = None
    
    # Pipeline options
    run_parallel: bool = False  # Run engines in parallel (not implemented yet)
    combine_results: bool = True
    save_combined_results: bool = True
    
    # Performance optimization
    skip_existing_results: bool = True

class PipelineEngine(BaseEngine):
    """Pipeline backtesting engine that runs multiple engines in sequence"""
    
    def __init__(self, config: PipelineEngineConfig = None):
        if config is None:
            config = PipelineEngineConfig()
        super().__init__(config)
        self.config = config
        
        # Initialize available engines
        self.available_engines = {
            'basic': BasicEngine,
            'risk_managed': RiskManagedEngine,
            'risk_managed_oos': RiskManagedEngine,
            'risk_managed_is': RiskManagedEngine,
            'statistical': StatisticalEngine,
            'walkforward': WalkforwardEngine,
            'alpha': AlphaEngine,
            'epoch': EpochEngine,
            'ml': MLEngine
        }
        
        # Set default engines to run if not specified
        if self.config.engines_to_run is None:
            # Keep default lineup unchanged (exclude variant aliases unless explicitly requested)
            self.config.engines_to_run = ['basic', 'risk_managed', 'statistical', 'walkforward', 'alpha', 'epoch', 'ml']
    
    def create_engine_instance(self, engine_name: str):
        """Create an instance of the specified engine with appropriate config"""
        if engine_name not in self.available_engines:
            raise ValueError(f"Unknown engine: {engine_name}")
        
        engine_class = self.available_engines[engine_name]
        
        # Get appropriate config for this engine
        config_attr = f"{engine_name}_engine_config"
        engine_config = getattr(self.config, config_attr, None)
        
        # Provide sensible defaults for variant aliases if not supplied
        if engine_config is None and engine_name in ("risk_managed_oos", "risk_managed_is"):
            selection = 'oos' if engine_name.endswith('oos') else 'is'
            engine_config = RiskManagedEngineConfig(
                walkforward_selection_criterion=selection,
                walkforward_window_mode='rolling',
                backtest_timeout=300,
                max_parameter_combinations=60,
                max_walkforward_windows=12,
                results_subdir_prefix=f"risk_managed_{selection}_backtest",
            )
        
        # Create engine instance
        if engine_config is None:
            engine_instance = engine_class()
        else:
            engine_instance = engine_class(engine_config)
        
        return engine_instance
    
    def run_engine(self, engine_name: str) -> List[Dict[str, Any]]:
        """Run a single engine and return its results"""
        try:
            self.logger.info(f"Starting {engine_name} engine...")
            
            # Create engine instance
            engine = self.create_engine_instance(engine_name)
            
            # Run the engine
            results = engine.run()
            
            self.logger.info(f"Completed {engine_name} engine with {len(results) if results else 0} results")
            return results if results else []
            
        except Exception as e:
            self.logger.error(f"Error running {engine_name} engine: {e}")
            return []
    
    def combine_engine_results(self, all_engine_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Combine results from all engines"""
        try:
            combined_results = {
                'pipeline_summary': {
                    'total_engines_run': len(all_engine_results),
                    'engines_run': list(all_engine_results.keys()),
                    'total_results': sum(len(results) for results in all_engine_results.values()),
                    'timestamp': datetime.now().isoformat()
                },
                'engine_results': all_engine_results,
                'cross_engine_analysis': {}
            }
            
            # Perform cross-engine analysis
            if len(all_engine_results) > 1:
                cross_analysis = self.perform_cross_engine_analysis(all_engine_results)
                combined_results['cross_engine_analysis'] = cross_analysis
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error combining engine results: {e}")
            return {}
    
    def perform_cross_engine_analysis(self, all_engine_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Perform analysis across different engines"""
        try:
            cross_analysis = {
                'strategy_consistency': {},
                'engine_correlation': {},
                'best_performing_engines': {}
            }
            
            # Analyze strategy consistency across engines
            strategy_performance = {}
            
            for engine_name, results in all_engine_results.items():
                for result in results:
                    strategy_name = result.get('strategy', 'unknown')
                    if strategy_name not in strategy_performance:
                        strategy_performance[strategy_name] = {}
                    
                    strategy_performance[strategy_name][engine_name] = {
                        'return_pct': result.get('return_pct', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'win_rate': result.get('win_rate', 0)
                    }
            
            # Calculate consistency metrics
            for strategy_name, performances in strategy_performance.items():
                if len(performances) > 1:
                    returns = [p.get('return_pct', 0) for p in performances.values()]
                    sharpe_ratios = [p.get('sharpe_ratio', 0) for p in performances.values()]
                    
                    consistency_score = 1 - (np.std(returns) / (np.mean(returns) + 1e-8))
                    
                    cross_analysis['strategy_consistency'][strategy_name] = {
                        'consistency_score': max(0, consistency_score),
                        'return_std': np.std(returns),
                        'sharpe_std': np.std(sharpe_ratios),
                        'engines_tested': len(performances)
                    }
            
            # Find best performing engines
            engine_performance = {}
            for engine_name, results in all_engine_results.items():
                if results:
                    avg_return = np.mean([r.get('return_pct', 0) for r in results])
                    avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in results])
                    avg_drawdown = np.mean([r.get('max_drawdown', 0) for r in results])
                    
                    engine_performance[engine_name] = {
                        'avg_return': avg_return,
                        'avg_sharpe': avg_sharpe,
                        'avg_drawdown': avg_drawdown,
                        'total_results': len(results)
                    }
            
            # Sort engines by performance
            sorted_engines = sorted(engine_performance.items(), 
                                  key=lambda x: x[1]['avg_return'], reverse=True)
            
            cross_analysis['best_performing_engines'] = {
                engine: perf for engine, perf in sorted_engines
            }
            
            return cross_analysis
            
        except Exception as e:
            self.logger.error(f"Error in cross-engine analysis: {e}")
            return {}
    
    def run(self):
        """Main function to run the pipeline"""
        self.logger.info(f"Starting pipeline with engines: {self.config.engines_to_run}")
        
        # Create results directory
        results_dir = self.create_results_directory("pipeline_backtest")
        
        # Run each engine
        all_engine_results = {}
        
        for engine_name in self.config.engines_to_run:
            if self.shutdown_requested:
                self.logger.info("Shutdown requested. Stopping pipeline.")
                break
            
            try:
                results = self.run_engine(engine_name)
                all_engine_results[engine_name] = results
                
            except Exception as e:
                self.logger.error(f"Failed to run {engine_name} engine: {e}")
                all_engine_results[engine_name] = []
        
        # Combine results if requested
        if self.config.combine_results and all_engine_results:
            try:
                combined_results = self.combine_engine_results(all_engine_results)
                
                if self.config.save_combined_results:
                    # Save combined results
                    combined_path = os.path.join(results_dir, 'combined_results')
                    self.save_results(combined_results, combined_path)
                    
                    # Save pipeline summary
                    pipeline_summary = {
                        'pipeline_config': {
                            'engines_run': self.config.engines_to_run,
                            'run_parallel': self.config.run_parallel,
                            'combine_results': self.config.combine_results
                        },
                        'execution_summary': {
                            'total_engines': len(self.config.engines_to_run),
                            'successful_engines': len([r for r in all_engine_results.values() if r]),
                            'total_results': sum(len(r) for r in all_engine_results.values()),
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    summary_path = os.path.join(results_dir, 'pipeline_summary')
                    self.save_results(pipeline_summary, summary_path)
                    
                    self.logger.info(f"Saved combined results to {results_dir}")
                
            except Exception as e:
                self.logger.error(f"Error combining results: {e}")
        
        # Save individual engine results
        for engine_name, results in all_engine_results.items():
            if results:
                engine_results_path = os.path.join(results_dir, f'{engine_name}_results')
                self.save_results(results, engine_results_path)
        
        if self.shutdown_requested:
            self.logger.info("Pipeline completed with early termination due to shutdown request.")
        else:
            self.logger.info("Pipeline completed successfully.")
        
        return all_engine_results
    
    def run_single_strategy_pipeline(self, strategy_path: str, data_file: str) -> Dict[str, Any]:
        """Run pipeline analysis on a single strategy-data combination"""
        try:
            strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
            data_name = os.path.splitext(os.path.basename(data_file))[0]
            
            self.logger.info(f"Running pipeline analysis for {strategy_name} on {data_name}")
            
            # Run each engine on this specific combination
            engine_results = {}
            
            for engine_name in self.config.engines_to_run:
                if self.shutdown_requested:
                    break
                
                try:
                    # Create engine instance
                    engine = self.create_engine_instance(engine_name)
                    
                    # Run single backtest
                    result = engine.run_single_backtest(strategy_path, data_file, "", data_name)
                    
                    if result:
                        engine_results[engine_name] = result
                    
                except Exception as e:
                    self.logger.error(f"Error running {engine_name} on {data_name}: {e}")
            
            # Combine results
            if engine_results:
                combined_result = {
                    'strategy': strategy_name,
                    'data_file': data_name,
                    'pipeline_results': engine_results,
                    'cross_engine_analysis': self.perform_cross_engine_analysis(engine_results)
                }
                
                return combined_result
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error in single strategy pipeline: {e}")
            return {}
    
    def run_targeted_pipeline(self, target_strategies: List[str] = None, 
                            target_data_files: List[str] = None) -> Dict[str, Any]:
        """Run pipeline on specific strategies and data files"""
        try:
            # Get target strategies and data files
            if target_strategies is None:
                strategies = self.discover_strategies()
            else:
                strategies = [s for s in self.discover_strategies() 
                           if any(target in s for target in target_strategies)]
            
            if target_data_files is None:
                data_files = self.discover_data_files()
            else:
                data_files = [d for d in self.discover_data_files() 
                            if any(target in d for target in target_data_files)]
            
            self.logger.info(f"Running targeted pipeline on {len(strategies)} strategies and {len(data_files)} data files")
            
            # Create results directory
            results_dir = self.create_results_directory("targeted_pipeline")
            
            # Run pipeline on each combination
            all_results = []
            
            for strategy_path in strategies:
                if self.shutdown_requested:
                    break
                
                strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
                self.logger.info(f"Processing strategy: {strategy_name}")
                
                strategy_results = []
                
                for data_file in data_files:
                    if self.shutdown_requested:
                        break
                    
                    data_name = os.path.splitext(os.path.basename(data_file))[0]
                    self.logger.info(f"Processing {data_name} for {strategy_name}")
                    
                    try:
                        result = self.run_single_strategy_pipeline(strategy_path, data_file)
                        if result:
                            strategy_results.append(result)
                            all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing {data_name}: {e}")
                
                # Save strategy-specific results
                if strategy_results:
                    strategy_path = os.path.join(results_dir, strategy_name)
                    self.save_results(strategy_results, strategy_path)
            
            # Save overall results
            if all_results:
                overall_path = os.path.join(results_dir, 'all_pipeline_results')
                self.save_results(all_results, overall_path)
                
                self.logger.info(f"Saved {len(all_results)} pipeline results to {results_dir}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in targeted pipeline: {e}")
            return [] 