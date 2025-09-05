"""
Epoch Backtesting Engine

Runs repeated backtests across multiple epochs (re-samples/shuffles) to assess
stability and robustness of strategy performance. This does not optimize
parameters; it evaluates robustness of the edge.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_engine import BaseEngine, EngineConfig

@dataclass
class EpochEngineConfig(EngineConfig):
    epochs: int = 20
    shuffle_within_blocks: bool = True
    block_size: int = 50
    parallel_workers: int = 2
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "epoch_backtest"
    skip_existing_results: bool = True

class EpochEngine(BaseEngine):
    def __init__(self, config: EpochEngineConfig = None):
        if config is None:
            config = EpochEngineConfig()
        super().__init__(config)
        self.config = config

    def _epoch_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.shuffle_within_blocks or self.config.block_size <= 1:
            return df.copy()
        try:
            n = len(df)
            blocks = []
            for start in range(0, n, self.config.block_size):
                end = min(start + self.config.block_size, n)
                block = df.iloc[start:end].copy()
                block = block.sample(frac=1.0, replace=False, random_state=np.random.randint(1_000_000))
                blocks.append(block)
            return pd.concat(blocks).sort_index()
        except Exception:
            return df.copy()

    def run_single_backtest(self, strategy_path: str, data_file: str, strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        try:
            strategy_cls = self.load_strategy(strategy_path)
            df = self.load_and_validate_data(data_file)

            # Run baseline
            baseline_stats = self.run_backtest(df, strategy_cls)
            if baseline_stats is None:
                return None

            # Run epochs
            epoch_returns: List[float] = []
            epoch_sharpes: List[float] = []
            epoch_drawdowns: List[float] = []
            for _ in range(int(self.config.epochs)):
                df_epoch = self._epoch_resample(df)
                stats = self.run_backtest(df_epoch, strategy_cls)
                if stats is None:
                    continue
                epoch_returns.append(float(stats.get('Return [%]', 0.0)))
                epoch_sharpes.append(float(stats.get('Sharpe Ratio', 0.0)))
                epoch_drawdowns.append(float(stats.get('Max. Drawdown [%]', 0.0)))

            # Aggregate
            def safe_mean(arr: List[float]) -> float:
                return float(np.mean(arr)) if arr else 0.0
            def safe_std(arr: List[float]) -> float:
                return float(np.std(arr)) if arr else 0.0

            result = {
                'strategy': os.path.basename(strategy_path),
                'data_file': data_name,
                'epochs': int(self.config.epochs),
                'baseline': {
                    'return_pct': baseline_stats.get('Return [%]', 0.0),
                    'sharpe_ratio': baseline_stats.get('Sharpe Ratio', 0.0),
                    'max_drawdown': baseline_stats.get('Max. Drawdown [%]', 0.0),
                },
                'epoch_summary': {
                    'avg_return': safe_mean(epoch_returns),
                    'std_return': safe_std(epoch_returns),
                    'avg_sharpe': safe_mean(epoch_sharpes),
                    'std_sharpe': safe_std(epoch_sharpes),
                    'avg_drawdown': safe_mean(epoch_drawdowns),
                    'std_drawdown': safe_std(epoch_drawdowns),
                    'n_successful_epochs': len(epoch_returns)
                },
                'epoch_results': {
                    'returns': epoch_returns,
                    'sharpes': epoch_sharpes,
                    'drawdowns': epoch_drawdowns
                }
            }

            # Save
            output_path = os.path.join(strat_result_dir, data_name)
            self.save_results(result, output_path)
            return result
        except Exception as e:
            self.logger.error(f"Error processing {data_name} with {os.path.basename(strategy_path)}: {e}")
            return None

    def check_existing_result(self, strat_result_dir: str, data_name: str) -> bool:
        if not self.config.skip_existing_results:
            return False
        json_path = os.path.join(strat_result_dir, f"{data_name}.json")
        return os.path.exists(json_path)

    def run(self):
        strategies = self.discover_strategies()

        results_root = Path(self.config.results_path)
        chosen_existing: Optional[Path] = None
        if self.config.prefer_existing_results_dir and results_root.exists():
            prefix = self.config.results_subdir_prefix
            cands = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}_")]
            if cands:
                chosen_existing = max(cands, key=lambda p: p.stat().st_mtime)
        if chosen_existing is not None:
            results_dir = str(chosen_existing)
            self.logger.info(f"Using existing results directory: {results_dir}")
        else:
            results_dir = self.create_results_directory(self.config.results_subdir_prefix)
            self.logger.info(f"Created new results directory: {results_dir}")

        all_results = []
        for strategy_path in strategies:
            if self.shutdown_requested:
                break
            strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
            self.logger.info(f"Processing strategy: {strategy_name}")

            strat_result_dir = os.path.join(results_dir, strategy_name)
            os.makedirs(strat_result_dir, exist_ok=True)

            data_files = self.discover_data_files_for_strategy(strategy_path)

            strategy_results: List[Dict[str, Any]] = []
            total_files = len(data_files)
            skipped_files = 0
            processed_files = 0

            files_to_process: List[Tuple[int, str, str]] = []
            for i, data_file in enumerate(data_files, 1):
                if self.shutdown_requested:
                    break
                data_name = os.path.splitext(os.path.basename(data_file))[0]
                if self.check_existing_result(strat_result_dir, data_name):
                    skipped_files += 1
                    continue
                files_to_process.append((i, data_file, data_name))

            if self.config.parallel_workers and self.config.parallel_workers > 1:
                self.logger.info(f"Running up to {self.config.parallel_workers} epoch evaluations in parallel for {strategy_name}")
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
                        except Exception as e:
                            self.logger.error(f"Error processing {data_name}: {e}")
            else:
                for (i, data_file, data_name) in files_to_process:
                    if self.shutdown_requested:
                        break
                    try:
                        result = self.run_single_backtest(strategy_path, data_file, strat_result_dir, data_name)
                        if result:
                            strategy_results.append(result)
                            all_results.append(result)
                            processed_files += 1
                    except Exception as e:
                        self.logger.error(f"Error processing {data_name}: {e}")

            self.logger.info(f"Completed {len(strategy_results)} epoch results for {strategy_name} (processed: {processed_files}, skipped: {skipped_files})")

        if all_results:
            summary_path = os.path.join(results_dir, 'all_results')
            self.save_results(all_results, summary_path)

        return all_results


