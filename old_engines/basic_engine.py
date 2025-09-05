"""
Basic Backtesting Engine

This engine provides standard backtesting functionality with significance testing
and quality assessment for strategy validation.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_engine import BaseEngine, EngineConfig

@dataclass
class BasicEngineConfig(EngineConfig):
    """Configuration for basic backtesting engine"""
    # Significance testing
    run_significance_tests: bool = True
    n_permutations: int = 100
    # Quality gate
    min_quality_score: float = 0.0
    
    # Performance optimization
    skip_existing_results: bool = True
    parallel_workers: int = 2

    # Results directory handling
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "basic_backtest"

class BasicEngine(BaseEngine):
    """Basic backtesting engine with significance testing"""
    
    def __init__(self, config: BasicEngineConfig = None):
        if config is None:
            config = BasicEngineConfig()
        super().__init__(config)
        self.config = config
    
    def create_random_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create random data for significance testing"""
        # Create random returns while preserving the original structure
        returns = original_data['Close'].pct_change().dropna()
        random_returns = np.random.choice(returns.values, size=len(returns), replace=True)
        
        # Reconstruct price series from random returns
        random_prices = [original_data['Close'].iloc[0]]
        for ret in random_returns:
            random_prices.append(random_prices[-1] * (1 + ret))
        
        # Create new dataframe with random data
        random_data = original_data.copy()
        random_data['Close'] = random_prices[:len(random_data)]
        
        # Recalculate other columns based on new close prices
        if 'High' in random_data.columns:
            random_data['High'] = random_data['Close'] * (1 + np.random.uniform(0, 0.02, len(random_data)))
        if 'Low' in random_data.columns:
            random_data['Low'] = random_data['Close'] * (1 - np.random.uniform(0, 0.02, len(random_data)))
        if 'Open' in random_data.columns:
            random_data['Open'] = random_data['Close'] * (1 + np.random.uniform(-0.01, 0.01, len(random_data)))
        
        return random_data
    
    def run_significance_test(self, original_stats: Dict[str, Any], strategy_cls: type, 
                            original_data: pd.DataFrame) -> Dict[str, Any]:
        """Run significance testing using permutation tests"""
        if not self.config.run_significance_tests:
            return {}
        
        try:
            # Run backtest on original data to get baseline
            original_return = original_stats.get('Return [%]', 0)
            original_sharpe = original_stats.get('Sharpe Ratio', 0)
            
            # Generate random data and test performance
            random_returns = []
            random_sharpes = []
            
            for i in range(self.config.n_permutations):
                try:
                    # Create random data
                    random_data = self.create_random_data(original_data)
                    
                    # Run backtest on random data
                    random_stats = self.run_backtest(random_data, strategy_cls)
                    if random_stats is not None:
                        random_returns.append(random_stats.get('Return [%]', 0))
                        random_sharpes.append(random_stats.get('Sharpe Ratio', 0))
                except Exception as e:
                    self.logger.warning(f"Error in permutation {i}: {e}")
                    continue
            
            if not random_returns:
                return {}
            
            # Calculate significance
            random_returns = np.array(random_returns)
            random_sharpes = np.array(random_sharpes)
            
            # Calculate p-values
            return_p_value = np.mean(random_returns >= original_return)
            sharpe_p_value = np.mean(random_sharpes >= original_sharpe)
            
            # Calculate confidence intervals
            return_ci = np.percentile(random_returns, [5, 95])
            sharpe_ci = np.percentile(random_sharpes, [5, 95])
            
            significance_results = {
                'original_return': original_return,
                'original_sharpe': original_sharpe,
                'random_returns_mean': np.mean(random_returns),
                'random_returns_std': np.std(random_returns),
                'random_sharpes_mean': np.mean(random_sharpes),
                'random_sharpes_std': np.std(random_sharpes),
                'return_p_value': return_p_value,
                'sharpe_p_value': sharpe_p_value,
                'return_significant': return_p_value < 0.05,
                'sharpe_significant': sharpe_p_value < 0.05,
                'return_confidence_interval': return_ci.tolist(),
                'sharpe_confidence_interval': sharpe_ci.tolist(),
                'n_permutations': len(random_returns)
            }
            
            return significance_results
            
        except Exception as e:
            self.logger.error(f"Error in significance testing: {e}")
            return {}
    
    def run_single_backtest(self, strategy_path: str, data_file: str, 
                           strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        """Run a single basic backtest and save results"""
        try:
            strategy_cls = self.load_strategy(strategy_path)
            df = self.load_and_validate_data(data_file)
            
            # Extract symbol and timeframe from filename
            symbol, timeframe = self.extract_info_from_filename(data_file)
            
            # Run backtest
            stats = self.run_backtest(df, strategy_cls)
            if stats is None:
                return None
            
            # Run significance testing
            significance_results = self.run_significance_test(stats, strategy_cls, df)
            
            # Assess quality
            quality_assessment = self.assess_quality(stats)
            passed_quality_threshold = (
                float(quality_assessment.get('score', 0)) >= float(getattr(self.config, 'min_quality_score', 0.0))
            )
            
            # Prepare result dictionary
            result_dict = {
                'strategy': os.path.basename(strategy_path),
                'data_file': data_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'total_trades': stats.get('# Trades', 0),
                'win_rate': stats.get('Win Rate [%]', 0.0),
                'profit_factor': stats.get('Profit Factor', 0.0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0.0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
                'return_pct': stats.get('Return [%]', 0.0),
                'volatility': stats.get('Volatility (Ann.) [%]', 0.0),
                'calmar_ratio': stats.get('Calmar Ratio', 0.0),
                'sortino_ratio': stats.get('Sortino Ratio', 0.0),
                'sqn': stats.get('SQN', 0.0),
                'equity_final': stats.get('Equity Final [$]', 0.0),
                'equity_peak': stats.get('Equity Peak [$]', 0.0),
                'buy_hold_return': stats.get('Buy & Hold Return [%]', 0.0),
                'exposure_time': stats.get('Exposure Time [%]', 0.0),
                'data_points': len(df),
                'quality_assessment': quality_assessment,
                'significance_testing': significance_results,
                'passed_quality_threshold': passed_quality_threshold
            }
            
            # Save individual result
            output_path = os.path.join(strat_result_dir, data_name)
            self.save_results(result_dict, output_path)
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error processing {data_name} with {os.path.basename(strategy_path)}: {e}")
            return None
    
    def check_existing_result(self, strat_result_dir: str, data_name: str) -> bool:
        """Check if a result file already exists for this data file"""
        if not self.config.skip_existing_results:
            return False
        
        json_path = os.path.join(strat_result_dir, f"{data_name}.json")
        return os.path.exists(json_path)
    
    def run(self):
        """Main function to run basic backtests"""
        # Discover strategies
        strategies = self.discover_strategies()
        # Informational count across all available data (not per-strategy filtered)
        try:
            total_data_files = len(self.discover_data_files())
        except Exception:
            total_data_files = 0
        self.logger.info(
            f"Starting basic backtesting with {len(strategies)} strategies and {total_data_files} data files (overall)."
        )
        
        # Choose results directory: prefer latest existing or create new
        results_root = Path(self.config.results_path)
        results_dir: str
        chosen_existing: Optional[Path] = None
        if self.config.prefer_existing_results_dir and results_root.exists():
            prefix = self.config.results_subdir_prefix
            candidates = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}_")]
            if candidates:
                chosen_existing = max(candidates, key=lambda p: p.stat().st_mtime)
        if chosen_existing is not None:
            results_dir = str(chosen_existing)
            self.logger.info(f"Using existing results directory: {results_dir}")
            resume_info = self.get_resume_info(results_dir)
            if resume_info['has_existing_results']:
                self.logger.info(f"Found {resume_info['total_results_processed']} existing results")
                self.logger.info(f"Last processed: {resume_info['last_processed_strategy']} - {resume_info['last_processed_data_file']}")
        else:
            results_dir = self.create_results_directory(self.config.results_subdir_prefix)
            self.logger.info(f"Created new results directory: {results_dir}")
        
        # Process each strategy
        all_results = []
        
        for strategy_path in strategies:
            if self.shutdown_requested:
                self.logger.info("Shutdown requested. Stopping backtest processing.")
                break
            
            strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
            self.logger.info(f"Processing strategy: {strategy_name}")
            
            # Create strategy-specific results directory
            strat_result_dir = os.path.join(results_dir, strategy_name)
            os.makedirs(strat_result_dir, exist_ok=True)
            
            # Discover data files specifically for this strategy
            data_files = self.discover_data_files_for_strategy(strategy_path)
            
            # Run backtests for strategy-specific data files (optionally in parallel)
            strategy_results: List[Dict[str, Any]] = []
            total_files = len(data_files)
            skipped_files = 0
            processed_files = 0

            # Prepare tasks list for files that need processing
            files_to_process: List[Tuple[int, str, str]] = []  # (idx, data_file, data_name)
            for i, data_file in enumerate(data_files, 1):
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested. Stopping backtest processing.")
                    break
                data_name = os.path.splitext(os.path.basename(data_file))[0]
                if self.check_existing_result(strat_result_dir, data_name):
                    self.logger.info(f"Skipping {data_name} ({i}/{total_files}) - result already exists")
                    skipped_files += 1
                    continue
                files_to_process.append((i, data_file, data_name))

            if self.config.parallel_workers and self.config.parallel_workers > 1:
                self.logger.info(f"Running up to {self.config.parallel_workers} backtests in parallel for {strategy_name}")
                with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                    future_to_meta = {
                        executor.submit(self.run_single_backtest, strategy_path, df_path, strat_result_dir, data_name): (idx, data_name)
                        for (idx, df_path, data_name) in files_to_process
                    }
                    for future in as_completed(future_to_meta):
                        idx, data_name = future_to_meta[future]
                        if self.shutdown_requested:
                            break
                        try:
                            result = future.result()
                            if result:
                                strategy_results.append(result)
                                all_results.append(result)
                                processed_files += 1
                                self.logger.info(f"Completed {data_name} ({idx}/{total_files}) for {strategy_name}")
                        except Exception as e:
                            self.logger.error(f"Error processing {data_name}: {e}")
            else:
                for (i, data_file, data_name) in files_to_process:
                    if self.shutdown_requested:
                        self.logger.info("Shutdown requested. Stopping backtest processing.")
                        break
                    self.logger.info(f"Processing {data_name} ({i}/{total_files}) for {strategy_name}")
                    try:
                        result = self.run_single_backtest(
                            strategy_path,
                            data_file,
                            strat_result_dir,
                            data_name
                        )
                        if result:
                            strategy_results.append(result)
                            all_results.append(result)
                            processed_files += 1
                    except Exception as e:
                        self.logger.error(f"Error processing {data_name}: {e}")
            
            if self.shutdown_requested:
                self.logger.info("Shutdown requested. Stopping strategy processing.")
                break
            
            self.logger.info(f"Completed {len(strategy_results)} backtests for {strategy_name} (processed: {processed_files}, skipped: {skipped_files})")
        
        # Save summary results
        if all_results:
            # Save all results as JSON
            summary_path = os.path.join(results_dir, 'all_results')
            self.save_results(all_results, summary_path)
            
            # Create basic backtest summary
            df = pd.DataFrame(all_results)
            basic_summary = {
                'total_backtests': len(all_results),
                'avg_return': df['return_pct'].mean(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'avg_drawdown': df['max_drawdown'].mean(),
                'avg_win_rate': df['win_rate'].mean(),
                'significance_tests_enabled': self.config.run_significance_tests,
                'n_permutations': self.config.n_permutations,
                'resume_info': self.get_resume_info(results_dir)
            }
            
            basic_summary_path = os.path.join(results_dir, 'basic_backtest_summary')
            self.save_results(basic_summary, basic_summary_path)
            
            self.logger.info(f"Saved {len(all_results)} total results to {results_dir}")
        else:
            self.logger.warning("No results to save")
        
        if self.shutdown_requested:
            self.logger.info("Backtest run completed with early termination due to shutdown request.")
        else:
            self.logger.info("Basic backtest run completed successfully.")
        
        return all_results 