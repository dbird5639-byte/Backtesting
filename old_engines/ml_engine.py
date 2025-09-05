"""
Machine Learning Backtesting Engine (lightweight)

Purpose: Evaluate learnability/edge via simple ML classifiers predicting next-bar
direction from engineered features. This complements strategy backtests by
assessing whether the data has predictive structure.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_engine import BaseEngine, EngineConfig

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

@dataclass
class MLEngineConfig(EngineConfig):
    test_size: float = 0.3
    random_state: int = 42
    use_random_forest: bool = True
    n_estimators: int = 200
    max_depth: Optional[int] = None
    parallel_workers: int = 2
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "ml_backtest"
    skip_existing_results: bool = True

class MLEngine(BaseEngine):
    def __init__(self, config: MLEngineConfig = None):
        if config is None:
            config = MLEngineConfig()
        super().__init__(config)
        self.config = config

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data['Close'].astype(float)
        data['ret_1'] = close.pct_change()
        data['ret_5'] = close.pct_change(5)
        data['ret_10'] = close.pct_change(10)
        data['sma_10'] = close.rolling(10).mean()
        data['sma_30'] = close.rolling(30).mean()
        data['sma_ratio'] = data['sma_10'] / (data['sma_30'] + 1e-12)
        data['vol_10'] = data['ret_1'].rolling(10).std()
        data['vol_30'] = data['ret_1'].rolling(30).std()
        data['mom_10'] = close.diff(10)
        data['range'] = (data['High'].astype(float) - data['Low'].astype(float)) / (data['Open'].astype(float) + 1e-12)
        data['volume_z'] = (data['Volume'].astype(float) - data['Volume'].astype(float).rolling(30).mean()) / (data['Volume'].astype(float).rolling(30).std() + 1e-12)
        data['target_up'] = (close.shift(-1) > close).astype(int)
        data = data.dropna()
        return data

    def _train_evaluate(self, features: pd.DataFrame) -> Dict[str, Any]:
        X = features.drop(columns=['target_up'])
        y = features['target_up']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, shuffle=False
        )
        if self.config.use_random_forest:
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                n_jobs=-1,
                random_state=self.config.random_state
            )
        else:
            model = LogisticRegression(max_iter=1000, n_jobs=None)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        pred = (proba >= 0.5).astype(int)
        metrics = {
            'auc': float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.5,
            'accuracy': float(accuracy_score(y_test, pred)),
            'precision': float(precision_score(y_test, pred, zero_division=0)),
            'recall': float(recall_score(y_test, pred, zero_division=0)),
        }
        return metrics

    def run_single_backtest(self, strategy_path: str, data_file: str, strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.error("scikit-learn not available. Install to use MLEngine.")
                return None

            df = self.load_and_validate_data(data_file)
            feat = self._build_features(df)
            if len(feat) < 200:
                return None
            metrics = self._train_evaluate(feat)
            result = {
                'strategy': os.path.basename(strategy_path),
                'data_file': data_name,
                'ml_metrics': metrics,
                'n_samples': int(len(feat))
            }
            output_path = os.path.join(strat_result_dir, data_name)
            self.save_results(result, output_path)
            return result
        except Exception as e:
            self.logger.error(f"Error processing {data_name} for ML engine: {e}")
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
                self.logger.info(f"Running up to {self.config.parallel_workers} ML evaluations in parallel for {strategy_name}")
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

        if all_results:
            summary_path = os.path.join(results_dir, 'all_results')
            self.save_results(all_results, summary_path)

        return all_results


