"""
Master Orchestrator Engine

This engine coordinates all other engines, applies market regime analysis to results,
and provides comprehensive visualization and analysis across all strategies, data files,
and engines. It serves as the central hub for the entire backtesting system.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.base.base_engine import BaseEngine, EngineConfig, BacktestResult
from core.base.base_strategy import BaseStrategy, StrategyConfig, Signal, Trade
from core.base.base_data_handler import BaseDataHandler, DataConfig
from core.base.base_risk_manager import BaseRiskManager, RiskConfig

# Import all engines
from core.engines.simple_engine import SimpleEngine, SimpleEngineConfig
from core.engines.advanced_engine import AdvancedEngine, AdvancedEngineConfig
from core.engines.permutation_engine import PermutationEngine, PermutationEngineConfig
from core.engines.risk_engine import RiskEngine, RiskEngineConfig
from core.engines.walkforward_engine import WalkForwardEngine, WalkForwardConfig
from core.engines.fibonacci_engine import FibonacciEngine, FibonacciConfig
from core.engines.regime_analysis_engine import RegimeAnalysisEngine, RegimeAnalysisConfig

warnings.filterwarnings('ignore')

@dataclass
class MasterOrchestratorConfig:
    """Configuration for master orchestrator engine"""
    # Engine configurations
    simple_engine_config: SimpleEngineConfig = field(default_factory=SimpleEngineConfig)
    advanced_engine_config: AdvancedEngineConfig = field(default_factory=AdvancedEngineConfig)
    permutation_engine_config: PermutationEngineConfig = field(default_factory=PermutationEngineConfig)
    risk_engine_config: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    walkforward_engine_config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    fibonacci_engine_config: FibonacciConfig = field(default_factory=FibonacciConfig)
    regime_analysis_config: RegimeAnalysisConfig = field(default_factory=RegimeAnalysisConfig)
    
    # Orchestration settings
    parallel_processing: bool = True
    max_workers: int = 4
    engine_timeout: int = 300  # seconds
    
    # Data processing
    data_directory: str = "./data"
    results_directory: str = "./results"
    cache_results: bool = True
    
    # Analysis settings
    generate_comprehensive_analysis: bool = True
    generate_cross_engine_comparison: bool = True
    generate_regime_based_analysis: bool = True
    generate_performance_heatmaps: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    generate_interactive_charts: bool = True
    export_to_excel: bool = True

@dataclass
class ComprehensiveAnalysisResult:
    """Results from comprehensive analysis across all engines"""
    timestamp: datetime
    data_files_processed: List[str]
    strategies_tested: List[str]
    engines_executed: List[str]
    
    # Engine results
    simple_engine_results: Dict[str, Any]
    advanced_engine_results: Dict[str, Any]
    permutation_engine_results: Dict[str, Any]
    risk_engine_results: Dict[str, Any]
    walkforward_engine_results: Dict[str, Any]
    fibonacci_engine_results: Dict[str, Any]
    
    # Regime analysis
    regime_analysis: Dict[str, Any]
    regime_based_performance: Dict[str, Any]
    
    # Cross-engine comparison
    cross_engine_comparison: Dict[str, Any]
    performance_rankings: Dict[str, List[str]]
    
    # Summary statistics
    summary_statistics: Dict[str, Any]

class MasterOrchestratorEngine(BaseEngine):
    """
    Master orchestrator engine that coordinates all other engines
    """
    
    def __init__(self, config: MasterOrchestratorConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        
        # Initialize all engines
        self.engines = {}
        self._initialize_engines()
        
        # Results storage
        self.comprehensive_results = []
        self.engine_results = {}
        self.regime_results = {}
        
    def _initialize_engines(self):
        """Initialize all available engines"""
        try:
            self.logger.info("Initializing engines...")
            
            # Initialize each engine
            self.engines['simple'] = SimpleEngine(self.config.simple_engine_config)
            self.engines['advanced'] = AdvancedEngine(self.config.advanced_engine_config)
            self.engines['permutation'] = PermutationEngine(self.config.permutation_engine_config)
            self.engines['risk'] = RiskEngine(self.config.risk_engine_config)
            self.engines['walkforward'] = WalkForwardEngine(self.config.walkforward_engine_config)
            self.engines['fibonacci'] = FibonacciEngine(self.config.fibonacci_engine_config)
            self.engines['regime_analysis'] = RegimeAnalysisEngine(self.config.regime_analysis_config)
            
            self.logger.info(f"Initialized {len(self.engines)} engines")
            
        except Exception as e:
            self.logger.error(f"Error initializing engines: {e}")
            raise
    
    def run_comprehensive_analysis(self, data_files: List[str], strategies: List[BaseStrategy]) -> ComprehensiveAnalysisResult:
        """Run comprehensive analysis across all engines and data files"""
        try:
            self.logger.info("Starting comprehensive analysis...")
            
            # First, run regime analysis on all data files
            regime_results = self._run_regime_analysis(data_files)
            
            # Run all engines on all data files and strategies
            engine_results = self._run_all_engines(data_files, strategies)
            
            # Apply regime analysis to engine results
            regime_based_performance = self._apply_regime_analysis_to_results(engine_results, regime_results)
            
            # Generate cross-engine comparison
            cross_engine_comparison = self._generate_cross_engine_comparison(engine_results)
            
            # Generate performance rankings
            performance_rankings = self._generate_performance_rankings(engine_results)
            
            # Generate summary statistics
            summary_statistics = self._generate_summary_statistics(engine_results, regime_results)
            
            # Create comprehensive result
            comprehensive_result = ComprehensiveAnalysisResult(
                timestamp=datetime.now(),
                data_files_processed=data_files,
                strategies_tested=[s.name for s in strategies],
                engines_executed=list(self.engines.keys()),
                
                simple_engine_results=engine_results.get('simple', {}),
                advanced_engine_results=engine_results.get('advanced', {}),
                permutation_engine_results=engine_results.get('permutation', {}),
                risk_engine_results=engine_results.get('risk', {}),
                walkforward_engine_results=engine_results.get('walkforward', {}),
                fibonacci_engine_results=engine_results.get('fibonacci', {}),
                
                regime_analysis=regime_results,
                regime_based_performance=regime_based_performance,
                
                cross_engine_comparison=cross_engine_comparison,
                performance_rankings=performance_rankings,
                
                summary_statistics=summary_statistics
            )
            
            self.comprehensive_results.append(comprehensive_result)
            self.engine_results = engine_results
            self.regime_results = regime_results
            
            self.logger.info("Comprehensive analysis completed successfully")
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    def _run_regime_analysis(self, data_files: List[str]) -> Dict[str, Any]:
        """Run regime analysis on all data files"""
        try:
            self.logger.info("Running regime analysis on all data files...")
            
            regime_engine = self.engines['regime_analysis']
            regime_results = {}
            
            for data_file in data_files:
                try:
                    self.logger.info(f"Analyzing regimes for: {data_file}")
                    
                    # Load data
                    data = regime_engine.load_data(data_file)
                    
                    # Detect regimes
                    file_regime_results = regime_engine.detect_market_regimes(data)
                    
                    # Store results
                    regime_results[data_file] = file_regime_results
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing regimes for {data_file}: {e}")
                    continue
            
            return regime_results
            
        except Exception as e:
            self.logger.error(f"Error in regime analysis: {e}")
            return {}
    
    def _run_all_engines(self, data_files: List[str], strategies: List[BaseStrategy]) -> Dict[str, Dict[str, Any]]:
        """Run all engines on all data files and strategies"""
        try:
            self.logger.info("Running all engines...")
            
            engine_results = {}
            
            # Run each engine
            for engine_name, engine in self.engines.items():
                if engine_name == 'regime_analysis':
                    continue  # Already handled separately
                
                self.logger.info(f"Running {engine_name} engine...")
                engine_results[engine_name] = self._run_engine_on_all_data(
                    engine, engine_name, data_files, strategies
                )
            
            return engine_results
            
        except Exception as e:
            self.logger.error(f"Error running all engines: {e}")
            return {}
    
    def _run_engine_on_all_data(self, engine: BaseEngine, engine_name: str, 
                                data_files: List[str], strategies: List[BaseStrategy]) -> Dict[str, Any]:
        """Run a specific engine on all data files and strategies"""
        try:
            engine_results = {}
            
            for data_file in data_files:
                try:
                    self.logger.info(f"Running {engine_name} on {data_file}")
                    
                    data = engine.load_data(data_file)
                    file_results = {}
                    
                    for strategy in strategies:
                        try:
                            # Run backtest based on engine type
                            if engine_name == 'walkforward':
                                result = engine.run_walkforward_analysis(data, strategy)
                            elif engine_name == 'fibonacci':
                                result = engine.run_comprehensive_analysis(data, strategy)
                            else:
                                result = engine.run_backtest(data, strategy)
                            
                            file_results[strategy.name] = result
                            
                        except Exception as e:
                            self.logger.error(f"Error running {engine_name} with {strategy.name} on {data_file}: {e}")
                            continue
                    
                    engine_results[data_file] = file_results
                    
                except Exception as e:
                    self.logger.error(f"Error running {engine_name} on {data_file}: {e}")
                    continue
            
            return engine_results
            
        except Exception as e:
            self.logger.error(f"Error running {engine_name} engine: {e}")
            return {}
    
    def _apply_regime_analysis_to_results(self, engine_results: Dict[str, Any], 
                                        regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime analysis to results from all engines"""
        try:
            self.logger.info("Applying regime analysis to engine results...")
            
            regime_engine = self.engines['regime_analysis']
            regime_based_performance = {}
            
            # Process each data file
            for data_file, file_regime_results in regime_results.items():
                if data_file not in regime_based_performance:
                    regime_based_performance[data_file] = {}
                
                # Process each engine's results for this data file
                for engine_name, engine_file_results in engine_results.items():
                    if data_file in engine_file_results:
                        # Apply regime analysis to this engine's results
                        regime_analysis = regime_engine.apply_regime_analysis_to_results(
                            list(engine_file_results[data_file].values()),
                            file_regime_results
                        )
                        
                        regime_based_performance[data_file][engine_name] = regime_analysis
            
            return regime_based_performance
            
        except Exception as e:
            self.logger.error(f"Error applying regime analysis: {e}")
            return {}
    
    def _generate_cross_engine_comparison(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-engine comparison analysis"""
        try:
            self.logger.info("Generating cross-engine comparison...")
            
            comparison = {
                'performance_comparison': {},
                'consistency_analysis': {},
                'risk_adjusted_metrics': {},
                'parameter_sensitivity': {}
            }
            
            # Compare performance across engines
            for data_file in self._get_common_data_files(engine_results):
                comparison['performance_comparison'][data_file] = self._compare_engine_performance(
                    engine_results, data_file
                )
            
            # Analyze consistency across engines
            comparison['consistency_analysis'] = self._analyze_cross_engine_consistency(engine_results)
            
            # Compare risk-adjusted metrics
            comparison['risk_adjusted_metrics'] = self._compare_risk_adjusted_metrics(engine_results)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error generating cross-engine comparison: {e}")
            return {}
    
    def _get_common_data_files(self, engine_results: Dict[str, Any]) -> List[str]:
        """Get data files that are common across all engines"""
        try:
            data_files_sets = []
            for engine_name, engine_results_data in engine_results.items():
                data_files_sets.append(set(engine_results_data.keys()))
            
            if not data_files_sets:
                return []
            
            # Find intersection
            common_files = set.intersection(*data_files_sets)
            return list(common_files)
            
        except Exception as e:
            self.logger.error(f"Error getting common data files: {e}")
            return []
    
    def _compare_engine_performance(self, engine_results: Dict[str, Any], data_file: str) -> Dict[str, Any]:
        """Compare performance of different engines on a specific data file"""
        try:
            comparison = {}
            
            for engine_name, engine_data in engine_results.items():
                if data_file in engine_data:
                    # Extract performance metrics
                    file_results = engine_data[data_file]
                    
                    # Aggregate metrics across strategies
                    aggregated_metrics = self._aggregate_metrics_across_strategies(file_results)
                    
                    comparison[engine_name] = aggregated_metrics
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing engine performance: {e}")
            return {}
    
    def _aggregate_metrics_across_strategies(self, file_results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate metrics across strategies for a single file"""
        try:
            aggregated = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'n_strategies': 0
            }
            
            for strategy_name, result in file_results.items():
                try:
                    # Extract metrics based on result type
                    if hasattr(result, 'total_return'):
                        aggregated['total_return'] += result.total_return
                        aggregated['sharpe_ratio'] += getattr(result, 'sharpe_ratio', 0)
                        aggregated['max_drawdown'] += getattr(result, 'max_drawdown', 0)
                        aggregated['win_rate'] += getattr(result, 'win_rate', 0)
                        aggregated['profit_factor'] += getattr(result, 'profit_factor', 0)
                    elif isinstance(result, dict):
                        aggregated['total_return'] += result.get('total_return', 0)
                        aggregated['sharpe_ratio'] += result.get('sharpe_ratio', 0)
                        aggregated['max_drawdown'] += result.get('max_drawdown', 0)
                        aggregated['win_rate'] += result.get('win_rate', 0)
                        aggregated['profit_factor'] += result.get('profit_factor', 0)
                    
                    aggregated['n_strategies'] += 1
                    
                except Exception as e:
                    continue
            
            # Calculate averages
            if aggregated['n_strategies'] > 0:
                for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']:
                    if key != 'n_strategies':
                        aggregated[key] /= aggregated['n_strategies']
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating metrics: {e}")
            return {}
    
    def _analyze_cross_engine_consistency(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of results across different engines"""
        try:
            consistency_analysis = {
                'return_correlation': {},
                'ranking_consistency': {},
                'outlier_analysis': {}
            }
            
            # Analyze return correlations across engines
            for data_file in self._get_common_data_files(engine_results):
                consistency_analysis['return_correlation'][data_file] = self._calculate_return_correlations(
                    engine_results, data_file
                )
            
            # Analyze ranking consistency
            consistency_analysis['ranking_consistency'] = self._analyze_ranking_consistency(engine_results)
            
            # Analyze outliers
            consistency_analysis['outlier_analysis'] = self._analyze_outliers(engine_results)
            
            return consistency_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-engine consistency: {e}")
            return {}
    
    def _calculate_return_correlations(self, engine_results: Dict[str, Any], data_file: str) -> Dict[str, float]:
        """Calculate return correlations between engines for a specific data file"""
        try:
            correlations = {}
            engine_returns = {}
            
            # Extract returns for each engine
            for engine_name, engine_data in engine_results.items():
                if data_file in engine_data:
                    file_results = engine_data[data_file]
                    returns = []
                    
                    for strategy_name, result in file_results.items():
                        try:
                            if hasattr(result, 'total_return'):
                                returns.append(result.total_return)
                            elif isinstance(result, dict):
                                returns.append(result.get('total_return', 0))
                        except:
                            continue
                    
                    if returns:
                        engine_returns[engine_name] = returns
            
            # Calculate correlations between engines
            engine_names = list(engine_returns.keys())
            for i, engine1 in enumerate(engine_names):
                for j, engine2 in enumerate(engine_names[i+1:], i+1):
                    if len(engine_returns[engine1]) == len(engine_returns[engine2]):
                        correlation = np.corrcoef(engine_returns[engine1], engine_returns[engine2])[0, 1]
                        correlations[f"{engine1}_vs_{engine2}"] = float(correlation) if not np.isnan(correlation) else 0.0
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating return correlations: {e}")
            return {}
    
    def _analyze_ranking_consistency(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of strategy rankings across engines"""
        try:
            ranking_consistency = {}
            
            # For each data file, analyze how consistently strategies are ranked
            for data_file in self._get_common_data_files(engine_results):
                strategy_rankings = {}
                
                # Get rankings from each engine
                for engine_name, engine_data in engine_results.items():
                    if data_file in engine_data:
                        file_results = engine_data[data_file]
                        
                        # Create ranking based on total return
                        rankings = []
                        for strategy_name, result in file_results.items():
                            try:
                                if hasattr(result, 'total_return'):
                                    rankings.append((strategy_name, result.total_return))
                                elif isinstance(result, dict):
                                    rankings.append((strategy_name, result.get('total_return', 0)))
                            except:
                                continue
                        
                        # Sort by return and create ranking
                        rankings.sort(key=lambda x: x[1], reverse=True)
                        strategy_rankings[engine_name] = [name for name, _ in rankings]
                
                # Calculate ranking consistency
                if len(strategy_rankings) > 1:
                    ranking_consistency[data_file] = self._calculate_ranking_consistency_score(strategy_rankings)
            
            return ranking_consistency
            
        except Exception as e:
            self.logger.error(f"Error analyzing ranking consistency: {e}")
            return {}
    
    def _calculate_ranking_consistency_score(self, strategy_rankings: Dict[str, List[str]]) -> float:
        """Calculate consistency score for strategy rankings across engines"""
        try:
            if len(strategy_rankings) < 2:
                return 1.0
            
            # Use Spearman's rank correlation
            engine_names = list(strategy_rankings.keys())
            consistency_scores = []
            
            for i, engine1 in enumerate(engine_names):
                for j, engine2 in enumerate(engine_names[i+1:], i+1):
                    ranking1 = strategy_rankings[engine1]
                    ranking2 = strategy_rankings[engine2]
                    
                    # Calculate Spearman correlation
                    if len(ranking1) == len(ranking2):
                        # Convert rankings to numerical values
                        rank1 = [ranking1.index(s) for s in ranking1]
                        rank2 = [ranking2.index(s) for s in ranking2]
                        
                        correlation = np.corrcoef(rank1, rank2)[0, 1]
                        if not np.isnan(correlation):
                            consistency_scores.append(correlation)
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ranking consistency score: {e}")
            return 0.0
    
    def _analyze_outliers(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze outliers in engine results"""
        try:
            outlier_analysis = {
                'extreme_returns': {},
                'inconsistent_performance': {},
                'anomaly_detection': {}
            }
            
            # Analyze extreme returns
            for data_file in self._get_common_data_files(engine_results):
                all_returns = []
                
                for engine_name, engine_data in engine_results.items():
                    if data_file in engine_data:
                        file_results = engine_data[data_file]
                        
                        for strategy_name, result in file_results.items():
                            try:
                                if hasattr(result, 'total_return'):
                                    all_returns.append(result.total_return)
                                elif isinstance(result, dict):
                                    all_returns.append(result.get('total_return', 0))
                            except:
                                continue
                
                if all_returns:
                    returns_array = np.array(all_returns)
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    
                    # Identify outliers (beyond 2 standard deviations)
                    outliers = returns_array[np.abs(returns_array - mean_return) > 2 * std_return]
                    
                    outlier_analysis['extreme_returns'][data_file] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': len(outliers) / len(returns_array) * 100,
                        'outlier_values': outliers.tolist() if len(outliers) > 0 else []
                    }
            
            return outlier_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing outliers: {e}")
            return {}
    
    def _compare_risk_adjusted_metrics(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare risk-adjusted metrics across engines"""
        try:
            risk_metrics = {
                'sharpe_ratios': {},
                'sortino_ratios': {},
                'calmar_ratios': {},
                'max_drawdowns': {}
            }
            
            # Extract risk-adjusted metrics for each engine
            for engine_name, engine_data in engine_results.items():
                risk_metrics['sharpe_ratios'][engine_name] = {}
                risk_metrics['sortino_ratios'][engine_name] = {}
                risk_metrics['calmar_ratios'][engine_name] = {}
                risk_metrics['max_drawdowns'][engine_name] = {}
                
                for data_file, file_results in engine_data.items():
                    file_metrics = {
                        'sharpe': [],
                        'sortino': [],
                        'calmar': [],
                        'max_dd': []
                    }
                    
                    for strategy_name, result in file_results.items():
                        try:
                            if hasattr(result, 'sharpe_ratio'):
                                file_metrics['sharpe'].append(result.sharpe_ratio)
                            if hasattr(result, 'sortino_ratio'):
                                file_metrics['sortino'].append(result.sortino_ratio)
                            if hasattr(result, 'calmar_ratio'):
                                file_metrics['calmar'].append(result.calmar_ratio)
                            if hasattr(result, 'max_drawdown'):
                                file_metrics['max_dd'].append(result.max_drawdown)
                        except:
                            continue
                    
                    # Calculate averages
                    risk_metrics['sharpe_ratios'][engine_name][data_file] = np.mean(file_metrics['sharpe']) if file_metrics['sharpe'] else 0.0
                    risk_metrics['sortino_ratios'][engine_name][data_file] = np.mean(file_metrics['sortino']) if file_metrics['sortino'] else 0.0
                    risk_metrics['calmar_ratios'][engine_name][data_file] = np.mean(file_metrics['calmar']) if file_metrics['calmar'] else 0.0
                    risk_metrics['max_drawdowns'][engine_name][data_file] = np.mean(file_metrics['max_dd']) if file_metrics['max_dd'] else 0.0
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error comparing risk-adjusted metrics: {e}")
            return {}
    
    def _generate_performance_rankings(self, engine_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate performance rankings across engines"""
        try:
            rankings = {
                'by_total_return': {},
                'by_sharpe_ratio': {},
                'by_consistency': {},
                'by_risk_adjusted': {}
            }
            
            # Generate rankings for each data file
            for data_file in self._get_common_data_files(engine_results):
                engine_performance = {}
                
                for engine_name, engine_data in engine_results.items():
                    if data_file in engine_data:
                        file_results = engine_data[data_file]
                        aggregated_metrics = self._aggregate_metrics_across_strategies(file_results)
                        
                        engine_performance[engine_name] = aggregated_metrics
                
                # Rank by total return
                return_rankings = sorted(engine_performance.items(), 
                                       key=lambda x: x[1].get('total_return', 0), reverse=True)
                rankings['by_total_return'][data_file] = [name for name, _ in return_rankings]
                
                # Rank by Sharpe ratio
                sharpe_rankings = sorted(engine_performance.items(), 
                                       key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)
                rankings['by_sharpe_ratio'][data_file] = [name for name, _ in sharpe_rankings]
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error generating performance rankings: {e}")
            return {}
    
    def _generate_summary_statistics(self, engine_results: Dict[str, Any], 
                                   regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        try:
            summary = {
                'total_analyses': 0,
                'successful_analyses': 0,
                'failed_analyses': 0,
                'average_performance': {},
                'best_performing_engines': {},
                'regime_summary': {},
                'data_coverage': {}
            }
            
            # Count total and successful analyses
            for engine_name, engine_data in engine_results.items():
                for data_file, file_results in engine_data.items():
                    summary['total_analyses'] += len(file_results)
                    summary['successful_analyses'] += sum(1 for r in file_results.values() if r is not None)
            
            summary['failed_analyses'] = summary['total_analyses'] - summary['successful_analyses']
            
            # Calculate average performance across engines
            summary['average_performance'] = self._calculate_average_performance(engine_results)
            
            # Identify best performing engines
            summary['best_performing_engines'] = self._identify_best_performing_engines(engine_results)
            
            # Summarize regime analysis
            summary['regime_summary'] = self._summarize_regime_analysis(regime_results)
            
            # Analyze data coverage
            summary['data_coverage'] = self._analyze_data_coverage(engine_results)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {e}")
            return {}
    
    def _calculate_average_performance(self, engine_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average performance across all engines"""
        try:
            avg_performance = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
            total_metrics = 0
            
            for engine_name, engine_data in engine_results.items():
                for data_file, file_results in engine_data.items():
                    for strategy_name, result in file_results.items():
                        try:
                            if hasattr(result, 'total_return'):
                                avg_performance['total_return'] += result.total_return
                                avg_performance['sharpe_ratio'] += getattr(result, 'sharpe_ratio', 0)
                                avg_performance['max_drawdown'] += getattr(result, 'max_drawdown', 0)
                                avg_performance['win_rate'] += getattr(result, 'win_rate', 0)
                                total_metrics += 1
                        except:
                            continue
            
            # Calculate averages
            if total_metrics > 0:
                for key in avg_performance:
                    avg_performance[key] /= total_metrics
            
            return avg_performance
            
        except Exception as e:
            self.logger.error(f"Error calculating average performance: {e}")
            return {}
    
    def _identify_best_performing_engines(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify best performing engines based on various metrics"""
        try:
            engine_scores = {}
            
            for engine_name, engine_data in engine_results.items():
                engine_score = 0
                total_tests = 0
                
                for data_file, file_results in engine_data.items():
                    for strategy_name, result in file_results.items():
                        try:
                            if hasattr(result, 'total_return'):
                                # Score based on return, Sharpe, and drawdown
                                score = 0
                                score += result.total_return * 100  # Return component
                                score += getattr(result, 'sharpe_ratio', 0) * 10  # Sharpe component
                                score -= getattr(result, 'max_drawdown', 0) * 50  # Drawdown penalty
                                
                                engine_score += score
                                total_tests += 1
                        except:
                            continue
                
                if total_tests > 0:
                    engine_scores[engine_name] = engine_score / total_tests
            
            # Rank engines by score
            ranked_engines = sorted(engine_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'rankings': [name for name, _ in ranked_engines],
                'scores': dict(ranked_engines)
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying best performing engines: {e}")
            return {}
    
    def _summarize_regime_analysis(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize regime analysis results"""
        try:
            summary = {
                'total_regimes_detected': 0,
                'regime_types': {},
                'average_regime_duration': 0.0,
                'regime_transitions': 0
            }
            
            total_duration = 0
            regime_count = 0
            
            for data_file, file_regime_results in regime_results.items():
                if 'regime_statistics' in file_regime_results:
                    regime_stats = file_regime_results['regime_statistics']
                    summary['total_regimes_detected'] += len(regime_stats)
                    
                    for regime_key, regime_data in regime_stats.items():
                        # Count regime types
                        regime_type = regime_data.get('characteristics', {}).get('regime_type', 'Unknown')
                        summary['regime_types'][regime_type] = summary['regime_types'].get(regime_type, 0) + 1
                        
                        # Accumulate duration
                        duration = regime_data.get('avg_duration_days', 0)
                        total_duration += duration
                        regime_count += 1
                
                if 'regime_transitions' in file_regime_results:
                    summary['regime_transitions'] += len(file_regime_results['regime_transitions'])
            
            # Calculate average duration
            if regime_count > 0:
                summary['average_regime_duration'] = total_duration / regime_count
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing regime analysis: {e}")
            return {}
    
    def _analyze_data_coverage(self, engine_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data coverage across engines"""
        try:
            coverage = {
                'total_data_files': 0,
                'files_per_engine': {},
                'common_files': [],
                'unique_files': []
            }
            
            # Collect all data files
            all_files = set()
            for engine_name, engine_data in engine_results.items():
                engine_files = set(engine_data.keys())
                all_files.update(engine_files)
                coverage['files_per_engine'][engine_name] = len(engine_files)
            
            coverage['total_data_files'] = len(all_files)
            
            # Find common files across all engines
            common_files = set.intersection(*[set(engine_data.keys()) for engine_data in engine_results.values()])
            coverage['common_files'] = list(common_files)
            
            # Find unique files (only used by one engine)
            engine_file_sets = [set(engine_data.keys()) for engine_data in engine_results.values()]
            unique_files = set()
            for i, engine_files in enumerate(engine_file_sets):
                other_engine_files = set.union(*engine_file_sets[:i] + engine_file_sets[i+1:])
                unique_files.update(engine_files - other_engine_files)
            
            coverage['unique_files'] = list(unique_files)
            
            return coverage
            
        except Exception as e:
            self.logger.error(f"Error analyzing data coverage: {e}")
            return {}
    
    def generate_comprehensive_visualizations(self, comprehensive_result: ComprehensiveAnalysisResult, 
                                           output_path: str = None) -> str:
        """Generate comprehensive visualizations for all analysis results"""
        try:
            if not self.config.generate_comprehensive_analysis:
                return ""
            
            if output_path is None:
                output_path = os.path.join(self.config.results_directory, 
                                         f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Generate cross-engine comparison charts
            if self.config.generate_cross_engine_comparison:
                self._generate_cross_engine_charts(comprehensive_result, output_path)
            
            # Generate regime-based analysis charts
            if self.config.generate_regime_based_analysis:
                self._generate_regime_based_charts(comprehensive_result, output_path)
            
            # Generate performance heatmaps
            if self.config.generate_performance_heatmaps:
                self._generate_performance_heatmaps(comprehensive_result, output_path)
            
            # Generate summary dashboard
            self._generate_summary_dashboard(comprehensive_result, output_path)
            
            self.logger.info(f"Comprehensive visualizations saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive visualizations: {e}")
            return ""
    
    def _generate_cross_engine_charts(self, comprehensive_result: ComprehensiveAnalysisResult, output_path: str):
        """Generate cross-engine comparison charts"""
        try:
            # Performance comparison across engines
            self._plot_engine_performance_comparison(comprehensive_result, output_path)
            
            # Consistency analysis
            self._plot_consistency_analysis(comprehensive_result, output_path)
            
            # Risk-adjusted metrics comparison
            self._plot_risk_metrics_comparison(comprehensive_result, output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating cross-engine charts: {e}")
    
    def _generate_regime_based_charts(self, comprehensive_result: ComprehensiveAnalysisResult, output_path: str):
        """Generate regime-based analysis charts"""
        try:
            # Regime performance analysis
            self._plot_regime_performance_analysis(comprehensive_result, output_path)
            
            # Regime transitions
            self._plot_regime_transitions(comprehensive_result, output_path)
            
            # Regime-based strategy performance
            self._plot_regime_strategy_performance(comprehensive_result, output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating regime-based charts: {e}")
    
    def _generate_performance_heatmaps(self, comprehensive_result: ComprehensiveAnalysisResult, output_path: str):
        """Generate performance heatmaps"""
        try:
            # Engine performance heatmap
            self._plot_engine_performance_heatmap(comprehensive_result, output_path)
            
            # Strategy performance heatmap
            self._plot_strategy_performance_heatmap(comprehensive_result, output_path)
            
            # Regime performance heatmap
            self._plot_regime_performance_heatmap(comprehensive_result, output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating performance heatmaps: {e}")
    
    def _generate_summary_dashboard(self, comprehensive_result: ComprehensiveAnalysisResult, output_path: str):
        """Generate summary dashboard"""
        try:
            # Create comprehensive dashboard
            self._plot_summary_dashboard(comprehensive_result, output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating summary dashboard: {e}")
    
    def save_comprehensive_results(self, comprehensive_result: ComprehensiveAnalysisResult, 
                                 output_path: str = None) -> str:
        """Save comprehensive analysis results"""
        try:
            if output_path is None:
                output_path = os.path.join(self.config.results_directory, 
                                         f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save main results
            results_file = os.path.join(output_path, "comprehensive_analysis_results.json")
            with open(results_file, 'w') as f:
                json.dump(self._comprehensive_result_to_dict(comprehensive_result), f, indent=2, default=str)
            
            # Save individual engine results
            engine_results_dir = os.path.join(output_path, "engine_results")
            os.makedirs(engine_results_dir, exist_ok=True)
            
            for engine_name, engine_data in self.engine_results.items():
                engine_file = os.path.join(engine_results_dir, f"{engine_name}_results.json")
                with open(engine_file, 'w') as f:
                    json.dump(engine_data, f, indent=2, default=str)
            
            # Save regime analysis results
            regime_file = os.path.join(output_path, "regime_analysis_results.json")
            with open(regime_file, 'w') as f:
                json.dump(self.regime_results, f, indent=2, default=str)
            
            # Generate visualizations
            if self.config.generate_comprehensive_analysis:
                self.generate_comprehensive_visualizations(comprehensive_result, output_path)
            
            self.logger.info(f"Comprehensive results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {e}")
            raise
    
    def _comprehensive_result_to_dict(self, comprehensive_result: ComprehensiveAnalysisResult) -> Dict[str, Any]:
        """Convert comprehensive result to dictionary for JSON serialization"""
        try:
            return {
                'timestamp': comprehensive_result.timestamp.isoformat(),
                'data_files_processed': comprehensive_result.data_files_processed,
                'strategies_tested': comprehensive_result.strategies_tested,
                'engines_executed': comprehensive_result.engines_executed,
                'summary_statistics': comprehensive_result.summary_statistics,
                'cross_engine_comparison': comprehensive_result.cross_engine_comparison,
                'performance_rankings': comprehensive_result.performance_rankings
            }
        except Exception as e:
            self.logger.error(f"Error converting comprehensive result to dict: {e}")
            return {}


# Simple strategy class for demonstration
class SimpleMAStrategy(BaseStrategy):
    """Simple moving average strategy for testing"""
    
    def __init__(self, name: str = "SimpleMA", parameters: Dict[str, Any] = None):
        super().__init__(name)
        self.parameters = parameters or {}
        self.short_window = self.parameters.get('short_window', 10)
        self.long_window = self.parameters.get('long_window', 20)
    
    def initialize(self, data: pd.DataFrame):
        """Initialize strategy with data"""
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        signals = pd.Series(0, index=data.index)
        
        if len(data) < self.long_window:
            return signals
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals[short_ma > long_ma] = 1   # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals
