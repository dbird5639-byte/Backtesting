"""
Statistical Backtesting Engine

This engine provides comprehensive statistical analysis including permutation tests,
bootstrap analysis, Monte Carlo simulations, and market regime detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils import resample

from base_engine import BaseEngine, EngineConfig
from utils.regime_utils import calculate_market_features as _calc_features, detect_regimes as _detect_regimes

@dataclass
class StatisticalEngineConfig(EngineConfig):
    """Configuration for statistical backtesting engine"""
    # Statistical testing parameters
    n_permutations: int = 50
    n_bootstrap_samples: int = 30
    n_monte_carlo_simulations: int = 25
    # GBM (Black–Scholes-like) Monte Carlo
    run_gbm_mc: bool = True
    gbm_simulations: int = 50
    gbm_annual_drift: Optional[float] = None
    gbm_annual_vol: Optional[float] = None
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    # Permutation test types
    run_permutation_tests: bool = True
    run_bootstrap_tests: bool = True
    run_monte_carlo_tests: bool = False
    
    # Market regime analysis
    enable_regime_analysis: bool = True
    regime_window_size: int = 50
    n_regimes: int = 3
    
    # Performance optimization
    skip_existing_results: bool = True
    parallel_workers: int = 2

    # Results directory handling
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "statistical_backtest"

class StatisticalEngine(BaseEngine):
    """Statistical backtesting engine with advanced analysis"""
    
    def __init__(self, config: StatisticalEngineConfig = None):
        if config is None:
            config = StatisticalEngineConfig()
        super().__init__(config)
        self.config = config
    
    def create_random_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create random data that preserves some market properties"""
        random_data = original_data.copy()
        
        # Preserve price levels but randomize returns
        returns = original_data['Close'].pct_change().dropna()
        
        # Bootstrap returns to preserve some market properties
        random_returns = resample(returns, n_samples=len(returns), 
                                random_state=np.random.randint(1000))
        
        # Reconstruct price series
        reconstructed_close = original_data['Close'].iloc[0] * (1 + np.array(random_returns)).cumprod()
        
        # Ensure the reconstructed series has the same length as original
        if len(reconstructed_close) < len(original_data):
            padding = [reconstructed_close[-1]] * (len(original_data) - len(reconstructed_close))
            reconstructed_close = np.concatenate([reconstructed_close, padding])
        elif len(reconstructed_close) > len(original_data):
            reconstructed_close = reconstructed_close[:len(original_data)]
        
        # Convert to pandas Series with proper index
        random_data['Close'] = pd.Series(reconstructed_close, index=random_data.index)
        
        # Adjust OHLC to maintain realistic relationships
        close_values = random_data['Close'].values
        high_values = np.zeros_like(close_values)
        low_values = np.zeros_like(close_values)
        open_values = np.zeros_like(close_values)
        
        # Set first values
        high_values[0] = close_values[0]
        low_values[0] = close_values[0]
        open_values[0] = close_values[0]
        
        # Generate realistic OHLC for remaining values
        for i in range(1, len(close_values)):
            prev_close = close_values[i-1]
            curr_close = close_values[i]
            
            # Generate realistic OHLC
            change = curr_close - prev_close
            high_offset = abs(change) * np.random.uniform(0, 0.5)
            low_offset = abs(change) * np.random.uniform(0, 0.5)
            
            high_values[i] = curr_close + high_offset
            low_values[i] = curr_close - low_offset
            open_values[i] = prev_close
        
        # Assign back to DataFrame
        random_data['High'] = high_values
        random_data['Low'] = low_values
        random_data['Open'] = open_values
        
        return random_data
    
    def market_preserving_permutation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Permute returns within volatility regimes (market-preserving)"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(20, min_periods=1).std()
        volatility_quantiles = pd.qcut(volatility.dropna(), q=5, labels=False, duplicates='drop')
        permuted_returns = pd.Series(0.0, index=data.index, dtype=float)
        
        for regime in volatility_quantiles.unique():
            if pd.isna(regime):
                continue
            regime_mask = volatility_quantiles == regime
            # Align mask to full index (skip first element because returns has NaN at start)
            idx = regime_mask.index
            mask_full = pd.Series(False, index=data.index)
            mask_full.loc[idx] = regime_mask.values
            # Extract returns for this regime (exclude first NaN)
            regime_returns = returns.loc[mask_full].dropna()
            if not regime_returns.empty:
                perm_values = np.random.permutation(regime_returns.to_numpy())
                permuted_returns.loc[regime_returns.index] = perm_values
        
        # Build price path aligned to data index with first return zero
        permuted_returns.iloc[0] = 0.0
        permuted_prices = float(data['Close'].iloc[0]) * (1 + permuted_returns).cumprod()
        permuted_data = data.copy()
        permuted_data['Close'] = permuted_prices
        return permuted_data
    
    def run_permutation_test(self, strategy_cls: type, df: pd.DataFrame) -> Dict[str, Any]:
        """Run permutation test to assess statistical significance"""
        try:
            # Run original strategy
            stats_original = self.run_backtest(df, strategy_cls)
            if stats_original is None:
                return {}
            
            # Run permutation tests
            random_performances = []
            max_attempts = min(self.config.n_permutations, 20)  # Limit for speed
            
            for i in range(max_attempts):
                try:
                    # Create random data
                    random_df = self.create_random_data(df)
                    
                    # Test strategy on random data
                    stats_random = self.run_backtest(random_df, strategy_cls)
                    if stats_random is not None:
                        random_performances.append(stats_random.get('Return [%]', 0.0))
                except Exception as e:
                    self.logger.warning(f"Random permutation test {i+1} failed: {e}")
                    continue
            
            if len(random_performances) < 5:
                self.logger.warning("Insufficient random results for permutation test")
                return {}
            
            # Calculate significance
            original_return = stats_original.get('Return [%]', 0.0)
            random_p_value = np.mean(np.array(random_performances) >= original_return)
            random_significant = random_p_value < self.config.significance_level
            
            # Calculate confidence intervals
            random_performances_array = np.array(random_performances)
            confidence_interval = np.percentile(random_performances_array, 
                                             [100 * (1 - self.config.confidence_level) / 2,
                                              100 * (1 + self.config.confidence_level) / 2])
            
            return {
                'original_core_metrics': self.select_core_metrics(stats_original),
                'random_performances': random_performances,
                'permutation_test': {
                    'p_value': random_p_value,
                    'significant': random_significant,
                    'original_return': original_return,
                    'random_mean': np.mean(random_performances),
                    'random_std': np.std(random_performances),
                    'confidence_interval': confidence_interval.tolist(),
                    'effect_size': (original_return - np.mean(random_performances)) / np.std(random_performances)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in permutation test: {e}")
            return {}
    
    def run_bootstrap_test(self, strategy_cls: type, df: pd.DataFrame) -> Dict[str, Any]:
        """Run bootstrap test to estimate confidence intervals"""
        try:
            # Run original strategy
            stats_original = self.run_backtest(df, strategy_cls)
            if stats_original is None:
                return {}
            
            # Bootstrap the data
            bootstrap_performances = []
            n_bootstrap = int(self.config.n_bootstrap_samples)
            
            for i in range(n_bootstrap):
                try:
                    # Bootstrap sample of the data
                    bootstrap_indices = np.random.choice(len(df), size=len(df), replace=True)
                    bootstrap_df = df.iloc[bootstrap_indices].copy()
                    bootstrap_df = bootstrap_df.sort_index()  # Maintain time order
                    
                    # Test strategy on bootstrap data
                    stats_bootstrap = self.run_backtest(bootstrap_df, strategy_cls)
                    if stats_bootstrap is not None:
                        bootstrap_performances.append(stats_bootstrap.get('Return [%]', 0.0))
                except Exception as e:
                    self.logger.warning(f"Bootstrap test {i+1} failed: {e}")
                    continue
            
            if len(bootstrap_performances) < 5:
                self.logger.warning("Insufficient bootstrap results")
                return {}
            
            # Calculate bootstrap statistics
            bootstrap_array = np.array(bootstrap_performances)
            bootstrap_ci = np.percentile(bootstrap_array, 
                                       [100 * (1 - self.config.confidence_level) / 2,
                                        100 * (1 + self.config.confidence_level) / 2])
            
            return {
                'bootstrap_test': {
                    'bootstrap_performances': bootstrap_performances,
                    'bootstrap_mean': np.mean(bootstrap_performances),
                    'bootstrap_std': np.std(bootstrap_performances),
                    'confidence_interval': bootstrap_ci.tolist(),
                    'original_return': stats_original.get('Return [%]', 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap test: {e}")
            return {}
    
    def run_monte_carlo_test(self, strategy_cls: type, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        try:
            # Run original strategy
            stats_original = self.run_backtest(df, strategy_cls)
            if stats_original is None:
                return {}
            
            # Monte Carlo simulation
            mc_performances = []
            n_simulations = int(self.config.n_monte_carlo_simulations)
            
            for i in range(n_simulations):
                try:
                    # Create market-preserving permutation
                    mc_df = self.market_preserving_permutation(df)
                    
                    # Test strategy on MC data
                    stats_mc = self.run_backtest(mc_df, strategy_cls)
                    if stats_mc is not None:
                        mc_performances.append(stats_mc.get('Return [%]', 0.0))
                except Exception as e:
                    self.logger.warning(f"Monte Carlo simulation {i+1} failed: {e}")
                    continue
            
            if len(mc_performances) < 5:
                self.logger.warning("Insufficient Monte Carlo results")
                return {}
            
            # Calculate MC statistics
            mc_array = np.array(mc_performances)
            mc_p_value = np.mean(mc_array >= stats_original.get('Return [%]', 0.0))
            
            return {
                'monte_carlo_test': {
                    'mc_performances': mc_performances,
                    'mc_mean': np.mean(mc_performances),
                    'mc_std': np.std(mc_performances),
                    'p_value': mc_p_value,
                    'significant': mc_p_value < self.config.significance_level,
                    'original_return': stats_original.get('Return [%]', 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo test: {e}")
            return {}

    def run_gbm_monte_carlo(self, strategy_cls: type, df: pd.DataFrame) -> Dict[str, Any]:
        """Run GBM (Black–Scholes-style) Monte Carlo to stress test strategy under synthetic paths.

        - Estimates drift/vol from data if not provided.
        - Generates price paths via geometric Brownian motion.
        - Caps simulations for speed.
        """
        try:
            if not getattr(self.config, 'run_gbm_mc', True):
                return {}
            close = df['Close'].astype(float)
            if len(close) < 30:
                return {}
            returns = close.pct_change().dropna()
            dt = 1/252
            mu = self.config.gbm_annual_drift if self.config.gbm_annual_drift is not None else returns.mean() * 252
            sigma = self.config.gbm_annual_vol if self.config.gbm_annual_vol is not None else returns.std() * np.sqrt(252)
            n_sims = int(getattr(self.config, 'gbm_simulations', 50))
            steps = len(df)
            start_price = float(close.iloc[0])

            gbm_performances: List[float] = []
            for i in range(n_sims):
                # Generate GBM path
                z = np.random.normal(size=steps-1)
                increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
                prices = np.empty(steps)
                prices[0] = start_price
                prices[1:] = start_price * np.exp(np.cumsum(increments))
                mc_df = df.copy()
                mc_df['Close'] = prices
                # Derive approximate OHLC consistent with Close
                high = prices * (1 + np.abs(np.random.normal(0, 0.002, size=steps)))
                low = prices * (1 - np.abs(np.random.normal(0, 0.002, size=steps)))
                open_ = np.r_[prices[0], prices[:-1]]
                mc_df['High'] = high
                mc_df['Low'] = low
                mc_df['Open'] = open_
                stats_mc = self.run_backtest(mc_df, strategy_cls)
                if stats_mc is not None:
                    gbm_performances.append(stats_mc.get('Return [%]', 0.0))

            if len(gbm_performances) < 5:
                return {}
            arr = np.array(gbm_performances)
            return {
                'gbm_monte_carlo': {
                    'mu_annual': float(mu),
                    'sigma_annual': float(sigma),
                    'simulations': n_sims,
                    'returns_mean': float(np.mean(arr)),
                    'returns_std': float(np.std(arr)),
                    'returns_p05': float(np.percentile(arr, 5)),
                    'returns_p95': float(np.percentile(arr, 95)),
                }
            }
        except Exception as e:
            self.logger.error(f"Error in GBM Monte Carlo: {e}")
            return {}
    
    def calculate_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return _calc_features(data)
    
    def detect_regimes(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
        clusters, regimes = _detect_regimes(features, self.config.n_regimes)
        return clusters, regimes
    
    def run_single_backtest(self, strategy_path: str, data_file: str, 
                           strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        """Run a single statistical backtest and save results"""
        try:
            strategy_cls = self.load_strategy(strategy_path)
            df = self.load_and_validate_data(data_file)
            
            # Extract symbol and timeframe from filename
            symbol, timeframe = self.extract_info_from_filename(data_file)
            
            # Run basic backtest
            stats = self.run_backtest(df, strategy_cls)
            if stats is None:
                return None
            
            # Run statistical tests
            statistical_tests = {}
            
            if self.config.run_permutation_tests:
                permutation_result = self.run_permutation_test(strategy_cls, df)
                statistical_tests.update(permutation_result)
            
            if self.config.run_bootstrap_tests:
                bootstrap_result = self.run_bootstrap_test(strategy_cls, df)
                statistical_tests.update(bootstrap_result)
            
            if self.config.run_monte_carlo_tests:
                mc_result = self.run_monte_carlo_test(strategy_cls, df)
                statistical_tests.update(mc_result)
            # Optional GBM MC
            gbm_result = self.run_gbm_monte_carlo(strategy_cls, df)
            statistical_tests.update(gbm_result)
            
            # Market regime analysis
            regime_analysis = {}
            if self.config.enable_regime_analysis:
                try:
                    features = self.calculate_market_features(df)
                    clusters, regimes = self.detect_regimes(features)
                    regime_analysis = {
                        'regimes': regimes,
                        'regime_distribution': np.bincount(clusters).tolist()
                    }
                except Exception as e:
                    self.logger.warning(f"Regime analysis failed: {e}")
            
            # Assess quality
            quality_assessment = self.assess_quality(stats)
            
            # Prepare result dictionary (use core metrics to avoid non-serializable content)
            result_dict = {
                'strategy': os.path.basename(strategy_path),
                'data_file': data_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'core_metrics': self.select_core_metrics(stats),
                'data_points': len(df),
                'quality_assessment': quality_assessment,
                'statistical_tests': statistical_tests,
                'regime_analysis': regime_analysis
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
        """Main function to run statistical backtests"""
        # Discover strategies
        strategies = self.discover_strategies()
        try:
            total_data_files = len(self.discover_data_files())
        except Exception:
            total_data_files = 0
        self.logger.info(f"Starting statistical backtesting with {len(strategies)} strategies and {total_data_files} data files (overall).")
        
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

            files_to_process: List[Tuple[int, str, str]] = []
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
            
            # Create statistical summary
            df = pd.DataFrame(all_results)
            statistical_summary = {
                'total_backtests': len(all_results),
                'avg_return': df['core_metrics'].apply(lambda x: x.get('Return [%]', 0) if isinstance(x, dict) else 0).mean(),
                'avg_sharpe': df['core_metrics'].apply(lambda x: x.get('Sharpe Ratio', 0) if isinstance(x, dict) else 0).mean(),
                'avg_drawdown': df['core_metrics'].apply(lambda x: x.get('Max. Drawdown [%]', 0) if isinstance(x, dict) else 0).mean(),
                'avg_win_rate': df['core_metrics'].apply(lambda x: x.get('Win Rate [%]', 0) if isinstance(x, dict) else 0).mean(),
                'statistical_tests_enabled': {
                    'permutation_tests': self.config.run_permutation_tests,
                    'bootstrap_tests': self.config.run_bootstrap_tests,
                    'monte_carlo_tests': self.config.run_monte_carlo_tests,
                    'regime_analysis': self.config.enable_regime_analysis
                },
                'test_parameters': {
                    'n_permutations': self.config.n_permutations,
                    'n_bootstrap_samples': self.config.n_bootstrap_samples,
                    'n_monte_carlo_simulations': self.config.n_monte_carlo_simulations,
                    'n_regimes': self.config.n_regimes
                },
                'resume_info': self.get_resume_info(results_dir)
            }
            
            statistical_summary_path = os.path.join(results_dir, 'statistical_summary')
            self.save_results(statistical_summary, statistical_summary_path)
            
            self.logger.info(f"Saved {len(all_results)} total results to {results_dir}")
        else:
            self.logger.warning("No results to save")
        
        if self.shutdown_requested:
            self.logger.info("Backtest run completed with early termination due to shutdown request.")
        else:
            self.logger.info("Statistical backtest run completed successfully.")
        
        return all_results 