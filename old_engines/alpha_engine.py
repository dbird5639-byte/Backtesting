"""
Alpha Detection Engine

This engine analyzes trading strategies to detect alpha, measure signal decay,
and assess signal strength over time.
"""

import os
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_engine import BaseEngine, EngineConfig

@dataclass
class AlphaEngineConfig(EngineConfig):
    """Configuration for alpha detection and decay engine"""
    # Alpha detection parameters
    min_alpha_threshold: float = 0.01
    window_size: int = 50
    significance_level: float = 0.05
    
    # Decay analysis parameters
    decay_window: int = 20
    min_signal_strength: float = 0.01
    
    # Signal analysis
    enable_signal_analysis: bool = True
    enable_decay_analysis: bool = True
    enable_alpha_detection: bool = True
    
    # Performance optimization
    skip_existing_results: bool = True
    parallel_workers: int = 2

    # Results directory handling
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "alpha_analysis"

class AlphaEngine(BaseEngine):
    """Alpha detection and decay analysis engine"""
    
    def __init__(self, config: AlphaEngineConfig = None):
        if config is None:
            config = AlphaEngineConfig()
        super().__init__(config)
        self.config = config
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal using erf."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def calculate_alpha_periods(self, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Detect periods where strategy generates significant alpha"""
        try:
            # Calculate rolling alpha (excess return)
            rolling_alpha = signals.rolling(window=self.config.window_size, min_periods=1).mean()
            
            # Calculate rolling significance
            rolling_std = returns.rolling(window=self.config.window_size, min_periods=1).std()
            denom = (rolling_std / np.sqrt(self.config.window_size)).replace(0, np.nan)
            rolling_t_stat = rolling_alpha / denom
            # p-value for two-tailed test using normal approximation
            rolling_p_value = 2 * (1 - rolling_t_stat.abs().apply(self._normal_cdf))
            
            # Identify significant alpha periods
            significant_periods = rolling_p_value < self.config.significance_level
            alpha_periods = significant_periods & (rolling_alpha > self.config.min_alpha_threshold)
            
            # Calculate alpha statistics
            alpha_stats = {
                'total_periods': len(alpha_periods),
                'significant_periods': alpha_periods.sum(),
                'alpha_coverage': alpha_periods.mean(),
                'avg_alpha': rolling_alpha[alpha_periods].mean() if alpha_periods.any() else 0,
                'max_alpha': rolling_alpha[alpha_periods].max() if alpha_periods.any() else 0,
                'alpha_volatility': rolling_alpha[alpha_periods].std() if alpha_periods.any() else 0
            }
            
            return {
                'alpha_periods': alpha_periods.tolist(),
                'rolling_alpha': rolling_alpha.tolist(),
                'rolling_p_value': rolling_p_value.tolist(),
                'alpha_stats': alpha_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in alpha period detection: {e}")
            return {}
    
    def detect_alpha_decay(self, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Detect alpha decay patterns"""
        try:
            # Calculate rolling alpha
            rolling_alpha = signals.rolling(window=self.config.window_size, min_periods=1).mean()
            
            # Detect decay periods (declining alpha)
            alpha_changes = rolling_alpha.diff()
            decay_periods = alpha_changes < -self.config.min_signal_strength
            
            # Calculate decay statistics
            avg_decay_rate = alpha_changes[decay_periods].mean() if decay_periods.any() else 0
            max_decay_rate = alpha_changes[decay_periods].min() if decay_periods.any() else 0
            decay_coverage = decay_periods.mean()
            # Simple decay score combining intensity and coverage
            decay_score = float(abs(avg_decay_rate)) * float(decay_coverage)
            decay_stats = {
                'total_decay_periods': decay_periods.sum(),
                'decay_coverage': decay_coverage,
                'avg_decay_rate': avg_decay_rate,
                'max_decay_rate': max_decay_rate,
                'decay_score': decay_score,
            }
            
            # Detect decay cycles
            decay_cycles = []
            in_decay = False
            cycle_start = None
            
            for i, is_decay in enumerate(decay_periods):
                if is_decay and not in_decay:
                    cycle_start = i
                    in_decay = True
                elif not is_decay and in_decay:
                    if cycle_start is not None:
                        decay_cycles.append({
                            'start': cycle_start,
                            'end': i,
                            'duration': i - cycle_start,
                            'decay_magnitude': alpha_changes.iloc[cycle_start:i].sum()
                        })
                    in_decay = False
            
            # Handle case where decay continues to end
            if in_decay and cycle_start is not None:
                decay_cycles.append({
                    'start': cycle_start,
                    'end': len(decay_periods) - 1,
                    'duration': len(decay_periods) - 1 - cycle_start,
                    'decay_magnitude': alpha_changes.iloc[cycle_start:].sum()
                })
            
            return {
                'decay_periods': decay_periods.tolist(),
                'alpha_changes': alpha_changes.tolist(),
                'decay_cycles': decay_cycles,
                'decay_stats': decay_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in alpha decay detection: {e}")
            return {}
    
    def analyze_signal_strength(self, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Analyze signal strength and consistency"""
        try:
            # Calculate signal strength metrics
            signal_correlation = signals.corr(returns)
            signal_autocorr = signals.autocorr()
            
            # Calculate signal-to-noise ratio
            signal_mean = signals.mean()
            signal_std = signals.std()
            signal_to_noise = signal_mean / signal_std if signal_std > 0 else 0
            avg_signal_strength = signals.abs().mean()
            
            # Calculate signal persistence
            signal_changes = signals.diff()
            signal_persistence = (signal_changes > 0).rolling(window=10).mean()
            
            # Calculate signal effectiveness
            positive_signals = signals > 0
            negative_signals = signals < 0
            positive_returns = returns > 0
            
            signal_effectiveness = {
                'positive_signal_accuracy': (positive_signals & positive_returns).sum() / positive_signals.sum() if positive_signals.sum() > 0 else 0,
                'negative_signal_accuracy': (negative_signals & ~positive_returns).sum() / negative_signals.sum() if negative_signals.sum() > 0 else 0,
                'overall_accuracy': ((positive_signals & positive_returns) | (negative_signals & ~positive_returns)).sum() / len(signals)
            }
            
            return {
                'signal_correlation': signal_correlation,
                'signal_autocorrelation': signal_autocorr,
                'signal_to_noise_ratio': signal_to_noise,
                'avg_signal_strength': avg_signal_strength,
                'signal_persistence': signal_persistence.tolist(),
                'signal_effectiveness': signal_effectiveness
            }
            
        except Exception as e:
            self.logger.error(f"Error in signal strength analysis: {e}")
            return {}
    
    def run_single_backtest(self, strategy_path: str, data_file: str, 
                           strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        """Run a single alpha analysis backtest and save results"""
        try:
            strategy_cls = self.load_strategy(strategy_path)
            df = self.load_and_validate_data(data_file)
            
            # Extract symbol and timeframe from filename
            symbol, timeframe = self.extract_info_from_filename(data_file)
            
            # Run basic backtest to get signals and returns
            stats = self.run_backtest(df, strategy_cls)
            if stats is None:
                return None
            
            # For alpha analysis, we need to simulate signals and returns
            # This is a simplified approach - in practice, you'd extract actual signals
            returns = df['Close'].pct_change().dropna()
            
            # Simulate strategy signals (this would be replaced with actual signal extraction)
            # For now, we'll use a simple moving average crossover as an example
            sma_short = df['Close'].rolling(10).mean()
            sma_long = df['Close'].rolling(30).mean()
            signals = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)
            signals = signals.dropna()
            
            # Align signals with returns
            common_index = signals.index.intersection(returns.index)
            signals = signals.loc[common_index]
            returns = returns.loc[common_index]
            
            # Run alpha analysis
            alpha_analysis = {}
            
            if self.config.enable_alpha_detection:
                alpha_periods = self.calculate_alpha_periods(signals, returns)
                alpha_analysis['alpha_detection'] = alpha_periods
            
            if self.config.enable_decay_analysis:
                decay_analysis = self.detect_alpha_decay(signals, returns)
                alpha_analysis['decay_analysis'] = decay_analysis
            
            if self.config.enable_signal_analysis:
                signal_analysis = self.analyze_signal_strength(signals, returns)
                alpha_analysis['signal_analysis'] = signal_analysis
            
            # Assess quality
            quality_assessment = self.assess_quality(stats)
            
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
                'alpha_analysis': alpha_analysis,
                'alpha_config': {
                    'min_alpha_threshold': self.config.min_alpha_threshold,
                    'window_size': self.config.window_size,
                    'significance_level': self.config.significance_level,
                    'decay_window': self.config.decay_window,
                    'min_signal_strength': self.config.min_signal_strength
                }
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
        """Main function to run alpha analysis backtests"""
        # Discover strategies
        strategies = self.discover_strategies()
        # Count overall data files safely (not per-strategy)
        try:
            total_data_files = len(self.discover_data_files())
        except Exception:
            total_data_files = 0
        self.logger.info(f"Starting alpha analysis with {len(strategies)} strategies and {total_data_files} data files.")
        
        # Choose results directory dynamically
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
                self.logger.info("Shutdown requested. Stopping alpha analysis.")
                break
            
            strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
            self.logger.info(f"Processing strategy: {strategy_name}")
            
            # Create strategy-specific results directory
            strat_result_dir = os.path.join(results_dir, strategy_name)
            os.makedirs(strat_result_dir, exist_ok=True)
            
            # Discover data files specifically for this strategy
            data_files = self.discover_data_files_for_strategy(strategy_path)
            
            # Run alpha analysis for strategy-specific data files (optionally in parallel)
            strategy_results: List[Dict[str, Any]] = []
            total_files = len(data_files)
            skipped_files = 0
            processed_files = 0

            files_to_process: List[Tuple[int, str, str]] = []
            for i, data_file in enumerate(data_files, 1):
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested. Stopping alpha analysis.")
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
                        self.logger.info("Shutdown requested. Stopping alpha analysis.")
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
            
            self.logger.info(f"Completed {len(strategy_results)} alpha analyses for {strategy_name} (processed: {processed_files}, skipped: {skipped_files})")
        
        # Save summary results
        if all_results:
            # Save all results as JSON
            summary_path = os.path.join(results_dir, 'all_results')
            self.save_results(all_results, summary_path)

            # Aggregate alpha analysis safely from dicts
            def _get_alpha_period_count(res: Dict[str, Any]) -> int:
                alpha_detection = (res.get('alpha_analysis') or {}).get('alpha_detection') or {}
                periods = alpha_detection.get('alpha_periods', [])
                return len(periods) if isinstance(periods, list) else 0

            def _get_decay_score(res: Dict[str, Any]) -> float:
                decay = (res.get('alpha_analysis') or {}).get('decay_analysis') or {}
                stats = decay.get('decay_stats') or {}
                return float(stats.get('decay_score', 0))

            def _get_avg_signal_strength(res: Dict[str, Any]) -> float:
                sig = (res.get('alpha_analysis') or {}).get('signal_analysis') or {}
                return float(sig.get('avg_signal_strength', 0))

            avg_alpha_periods = float(np.mean([_get_alpha_period_count(r) for r in all_results])) if all_results else 0.0
            avg_decay_score = float(np.mean([_get_decay_score(r) for r in all_results])) if all_results else 0.0
            avg_signal_strength = float(np.mean([_get_avg_signal_strength(r) for r in all_results])) if all_results else 0.0

            alpha_summary = {
                'total_analyses': len(all_results),
                'avg_alpha_periods': avg_alpha_periods,
                'avg_decay_score': avg_decay_score,
                'avg_signal_strength': avg_signal_strength,
                'alpha_config': {
                    'min_alpha_threshold': self.config.min_alpha_threshold,
                    'window_size': self.config.window_size,
                    'significance_level': self.config.significance_level,
                    'decay_window': self.config.decay_window,
                    'min_signal_strength': self.config.min_signal_strength,
                    'enable_signal_analysis': self.config.enable_signal_analysis,
                    'enable_decay_analysis': self.config.enable_decay_analysis,
                    'enable_alpha_detection': self.config.enable_alpha_detection,
                },
                'resume_info': self.get_resume_info(results_dir)
            }

            alpha_summary_path = os.path.join(results_dir, 'alpha_analysis_summary')
            self.save_results(alpha_summary, alpha_summary_path)

            self.logger.info(f"Saved {len(all_results)} total results to {results_dir}")
        else:
            self.logger.warning("No results to save")
        
        if self.shutdown_requested:
            self.logger.info("Alpha analysis run completed with early termination due to shutdown request.")
        else:
            self.logger.info("Alpha analysis backtest run completed successfully.")
        
        return all_results 