#!/usr/bin/env python3
"""
Optimized Core Engine - High performance, reliability, and comprehensive results
Based on base_optimized_engine with enhanced features
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from backtesting import Backtest
import importlib.util
import inspect

# Import base optimized engine
from .base_optimized_engine import BaseOptimizedEngine, BaseEngineConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="A contingent SL/TP order would execute in the same bar*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backtesting\._stats")
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide", category=RuntimeWarning)

@dataclass
class EngineConfig(BaseEngineConfig):
    """Configuration for optimized core engine"""
    # Core-specific parameters
    quality_threshold: float = 0.0
    run_significance_tests: bool = True
    n_permutations: int = 50
    
    # Performance optimization
    parallel_workers: int = 4
    batch_size: int = 10
    
    # Results management
    save_individual_results: bool = True
    save_combined_results: bool = True
    save_summary: bool = True
    
    # Terminal output
    verbose: bool = True
    progress_interval: int = 5
    show_performance_stats: bool = True

class CoreEngine(BaseOptimizedEngine):
    """Optimized core engine with high performance and comprehensive results"""
    
    def __init__(self, config: EngineConfig = None):
        super().__init__(config or EngineConfig(), "CoreEngine")
        self.config = self.config  # Type hint for IDE
    
    def calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score for strategy performance"""
        score = 0.0
        
        # Return quality (40% weight)
        total_return = result.get('total_return', 0)
        if total_return > 0.2:  # >20% return
            score += 4.0
        elif total_return > 0.1:  # >10% return
            score += 3.0
        elif total_return > 0.05:  # >5% return
            score += 2.0
        elif total_return > 0:  # Positive return
            score += 1.0
        
        # Sharpe ratio quality (30% weight)
        sharpe_ratio = result.get('sharpe_ratio', 0)
        if sharpe_ratio > 2.0:
            score += 3.0
        elif sharpe_ratio > 1.5:
            score += 2.5
        elif sharpe_ratio > 1.0:
            score += 2.0
        elif sharpe_ratio > 0.5:
            score += 1.0
        
        # Drawdown quality (20% weight)
        max_drawdown = result.get('max_drawdown', 1.0)
        if max_drawdown < 0.05:  # <5% drawdown
            score += 2.0
        elif max_drawdown < 0.1:  # <10% drawdown
            score += 1.5
        elif max_drawdown < 0.2:  # <20% drawdown
            score += 1.0
        
        # Win rate quality (10% weight)
        win_rate = result.get('win_rate', 0)
        if win_rate > 0.7:  # >70% win rate
            score += 1.0
        elif win_rate > 0.6:  # >60% win rate
            score += 0.8
        elif win_rate > 0.5:  # >50% win rate
            score += 0.5
        
        return score
    
    def run_significance_test(self, original_result: Dict[str, Any], 
                            strategy_cls, data: pd.DataFrame) -> Dict[str, Any]:
        """Run significance testing using permutation tests"""
        if not self.config.run_significance_tests:
            return {}
        
        try:
            original_return = original_result['total_return']
            random_returns = []
            
            self.logger.debug(f"Running {self.config.n_permutations} permutations...")
            
            for i in range(self.config.n_permutations):
                if self.interrupted:
                    break
                
                # Create random data by shuffling returns
                random_data = data.copy()
                returns = data['Close'].pct_change().dropna()
                shuffled_returns = np.random.permutation(returns.values)
                
                # Reconstruct price series
                random_prices = [data['Close'].iloc[0]]
                for ret in shuffled_returns:
                    random_prices.append(random_prices[-1] * (1 + ret))
                
                random_data['Close'] = random_prices[:len(random_data)]
                
                # Run backtest on random data
                random_result = self.run_single_backtest(strategy_cls(), random_data)
                if random_result:
                    random_returns.append(random_result['total_return'])
            
            if random_returns:
                random_returns = np.array(random_returns)
                p_value = np.mean(random_returns >= original_return)
                
                return {
                    'significance_test': {
                        'p_value': p_value,
                        'is_significant': p_value < 0.05,
                        'random_mean_return': np.mean(random_returns),
                        'random_std_return': np.std(random_returns),
                        'n_permutations': len(random_returns),
                        'original_return': original_return
                    }
                }
            
        except Exception as e:
            self.logger.error(f"Error in significance test: {e}")
        
        return {}
    
    def process_file_strategy_combination(self, data_file: Path, strategy_file: Path) -> List[Dict[str, Any]]:
        """Process a single data file with a single strategy"""
        results = []
        
        try:
            # Load data
            data = self.load_data(data_file)
            if data is None:
                return results
            
            # Load strategy
            strategy_cls = self.load_strategy(strategy_file)
            if strategy_cls is None:
                return results
            
            strategy_name = strategy_file.stem
            data_file_name = data_file.stem
            
            # Run backtest
            result = self.run_single_backtest(strategy_cls(), data)
            if result is None:
                return results
            
            # Calculate quality score
            result['quality_score'] = self.calculate_quality_score(result)
            
            # Filter by quality threshold
            if result['quality_score'] < self.config.quality_threshold:
                self.logger.debug(f"Filtered out {strategy_name} on {data_file_name} "
                                f"(quality score: {result['quality_score']:.2f})")
                return results
            
            # Run significance test
            significance_results = self.run_significance_test(result, strategy_cls, data)
            result.update(significance_results)
            
            # Add metadata
            result.update({
                'strategy_name': strategy_name,
                'data_file': data_file_name,
                'engine_name': 'CoreEngine',
                'timestamp': datetime.now().isoformat()
            })
            
            results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error processing {strategy_file.name} on {data_file.name}: {e}")
        
        return results
    
    def run(self):
        """Main execution method"""
        # Discover files
        data_files = self.discover_data_files()
        strategy_files = self.discover_strategy_files()
        
        if not data_files or not strategy_files:
            self.logger.error("❌ No data files or strategy files found")
            return
        
        # Process combinations
        all_results = self.process_combinations(data_files, strategy_files)
        
        # Save results
        self.save_results(all_results)
        
        # Final summary
        if all_results:
            df = pd.DataFrame(all_results)
            self.logger.info(f"📊 Final Summary:")
            self.logger.info(f"   • Total combinations: {len(all_results)}")
            self.logger.info(f"   • Strategies tested: {df['strategy_name'].nunique()}")
            self.logger.info(f"   • Data files tested: {df['data_file'].nunique()}")
            self.logger.info(f"   • Average quality score: {df['quality_score'].mean():.2f}")
            self.logger.info(f"   • Best return: {df['total_return'].max():.2%}")
            self.logger.info(f"   • Best Sharpe ratio: {df['sharpe_ratio'].max():.2f}")
            
            # Significance test summary
            if 'significance_test.p_value' in df.columns:
                significant_count = df[df['significance_test.is_significant'] == True].shape[0]
                self.logger.info(f"   • Significant strategies: {significant_count}/{len(all_results)} "
                               f"({significant_count/len(all_results)*100:.1f}%)")

def main():
    """Main entry point"""
    config = EngineConfig()
    engine = CoreEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
