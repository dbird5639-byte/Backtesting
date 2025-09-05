"""
Neural Network Engine (optional, lightweight)

Trains a small feedforward network to predict next-bar direction from
engineered features to assess non-linear learnability/edge.
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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

@dataclass
class NNEngineConfig(EngineConfig):
    test_size: float = 0.3
    random_state: int = 42
    hidden_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    parallel_workers: int = 1
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "nn_backtest"
    skip_existing_results: bool = True

if TORCH_AVAILABLE:
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim: int, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
        def forward(self, x):
            return self.net(x)
else:
    # Define a lightweight placeholder so module import does not fail when torch is missing
    class SimpleMLP:  # type: ignore
        pass

class NNEngine(BaseEngine):
    def __init__(self, config: NNEngineConfig = None):
        if config is None:
            config = NNEngineConfig()
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
        data['target_up'] = (close.shift(-1) > close).astype(int)
        data = data.dropna()
        return data

    def run_single_backtest(self, strategy_path: str, data_file: str, strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        try:
            if not TORCH_AVAILABLE:
                self.logger.error("PyTorch not available. Install to use NNEngine.")
                return None
            df = self.load_and_validate_data(data_file)
            feat = self._build_features(df)
            if len(feat) < 500:
                return None
            y = feat['target_up'].values.astype(np.float32)
            X = feat.drop(columns=['target_up']).values.astype(np.float32)
            split = int(len(X) * (1 - float(self.config.test_size)))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            model = SimpleMLP(input_dim=X.shape[1], hidden=int(self.config.hidden_size))
            optimizer = optim.Adam(model.parameters(), lr=float(self.config.lr))
            criterion = nn.BCELoss()
            model.train()
            X_train_t = torch.from_numpy(X_train)
            y_train_t = torch.from_numpy(y_train).view(-1, 1)
            for _ in range(int(self.config.epochs)):
                optimizer.zero_grad()
                pred = model(X_train_t)
                loss = criterion(pred, y_train_t)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                proba = model(torch.from_numpy(X_test)).numpy().reshape(-1)
            pred_label = (proba >= 0.5).astype(int)
            acc = float((pred_label == y_test).mean())
            result = {
                'strategy': os.path.basename(strategy_path),
                'data_file': data_name,
                'nn_metrics': {
                    'accuracy': acc,
                    'test_size': int(len(y_test)),
                    'epochs': int(self.config.epochs),
                    'hidden_size': int(self.config.hidden_size),
                },
            }
            output_path = os.path.join(strat_result_dir, data_name)
            self.save_results(result, output_path)
            return result
        except Exception as e:
            self.logger.error(f"Error processing {data_name} for NN engine: {e}")
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
            files_to_process: List[Tuple[int, str, str]] = []
            for i, data_file in enumerate(data_files, 1):
                data_name = os.path.splitext(os.path.basename(data_file))[0]
                if self.check_existing_result(strat_result_dir, data_name):
                    continue
                files_to_process.append((i, data_file, data_name))
            for (i, data_file, data_name) in files_to_process:
                if self.shutdown_requested:
                    break
                res = self.run_single_backtest(strategy_path, data_file, strat_result_dir, data_name)
                if res:
                    all_results.append(res)
        if all_results:
            self.save_results(all_results, os.path.join(results_dir, 'all_results'))
        return all_results


