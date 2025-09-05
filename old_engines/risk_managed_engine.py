"""
Risk-Managed Backtesting Engine

This engine provides advanced risk management features including position sizing,
stop losses, take profits, trailing stops, and drawdown protection.
It also includes walkforward optimization for risk parameters.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from itertools import product
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_engine import BaseEngine, EngineConfig
from utils.risk_utils import calculate_risk_metrics as _calc_risk

@dataclass
class RiskManagedEngineConfig(EngineConfig):
    """Configuration for risk-managed backtesting engine"""
    # Risk management parameters
    stop_loss_pct: float = 0.25  # 25%
    take_profit_pct: float = 0.50  # 50%
    trailing_stop_pct: float = 0.25  # 25% (matched to SL)
    position_size_pct: float = 0.02  # 2% of equity per trade
    max_consecutive_losses: int = 5
    max_drawdown_pct: float = 0.20  # 20%
    
    # Risk management features
    enable_position_sizing: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_trailing_stop: bool = True
    enable_drawdown_protection: bool = True
    enable_consecutive_loss_protection: bool = True
    
    # Walkforward optimization for risk parameters
    enable_walkforward_optimization: bool = True
    # Report-only mode: do not auto-select or apply any "best" parameters; only evaluate and report
    report_only: bool = True
    # Reporting controls
    report_top_k: int = 5
    report_selection_metric: str = "oos_return"  # or "oos_sharpe"
    # Default walkforward split ~70/30 with step equal to test size
    walkforward_train_size: int = 700
    walkforward_test_size: int = 300
    walkforward_step_size: int = 300
    # Minimum total candles required to run walkforward optimization (allows dynamic downscaling)
    walkforward_min_total_size: int = 300
    # Walkforward selection behavior
    # Options: 'oos' (select by out-of-sample only), 'is' (select by in-sample only), 'combined' (blend IS/OOS)
    walkforward_selection_criterion: str = "combined"
    # Windowing: 'rolling' uses fixed-length sliding windows; 'expanding' grows the train window each step
    walkforward_window_mode: str = "rolling"
    
    # Parameter optimization ranges
    stop_loss_range: List[float] = None  # Will be set to [0.01, 0.02, 0.03, 0.05] if None
    take_profit_range: List[float] = None  # Will be set to [0.02, 0.04, 0.06, 0.08] if None
    position_size_range: List[float] = None  # Will be set to [0.05, 0.10, 0.15, 0.20] if None
    trailing_stop_range: List[float] = None  # Will be set to [0.01, 0.015, 0.02, 0.025] if None
    
    # Performance optimization
    skip_existing_results: bool = True
    parallel_workers: int = 2
    max_parameter_combinations: int = 80
    max_walkforward_windows: int = 20
    save_full_parameter_eval_csv: bool = True

    # Results directory handling
    prefer_existing_results_dir: bool = True
    results_subdir_prefix: str = "risk_managed_backtest"

@dataclass
class RiskParameterSet:
    """A set of risk parameters for optimization"""
    stop_loss_pct: float
    take_profit_pct: float
    position_size_pct: float
    trailing_stop_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'position_size_pct': self.position_size_pct,
            'trailing_stop_pct': self.trailing_stop_pct
        }

@dataclass
class OptimizationResult:
    """Result from parameter optimization"""
    parameter_set: RiskParameterSet
    train_performance: Dict[str, Any]
    test_performance: Dict[str, Any]
    optimization_score: float
    regime: str = "unknown"
    window_index: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameter_set': self.parameter_set.to_dict(),
            'train_performance': self.train_performance,
            'test_performance': self.test_performance,
            'optimization_score': self.optimization_score
        }

class RiskManagedEngine(BaseEngine):
    """Risk-managed backtesting engine with advanced risk controls and walkforward optimization"""
    
    def __init__(self, config: RiskManagedEngineConfig = None):
        if config is None:
            config = RiskManagedEngineConfig()
        super().__init__(config)
        self.config = config
        
        # Set default parameter ranges if not provided
        if self.config.stop_loss_range is None:
            # Focus on practical lows (5%, 10%) and high-tolerance levels (50%, 75%, 99%)
            self.config.stop_loss_range = [0.05, 0.10, 0.50, 0.75, 0.99]
        if self.config.take_profit_range is None:
            self.config.take_profit_range = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        if self.config.position_size_range is None:
            self.config.position_size_range = [0.01, 0.02, 0.05, 0.10]
        if self.config.trailing_stop_range is None:
            # Mirror SL emphasis: 5%, 10%, and high-tolerance trailing stops
            self.config.trailing_stop_range = [0.05, 0.10, 0.50, 0.75, 0.99]
    
    def apply_risk_management_to_strategy(self, strategy_cls: type, params: RiskParameterSet = None) -> type:
        """Apply risk management parameters to strategy class"""
        if params is None:
            # Use default config parameters
            params = RiskParameterSet(
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                position_size_pct=self.config.position_size_pct,
                trailing_stop_pct=self.config.trailing_stop_pct
            )
        
        # Convert from decimal (e.g., 0.05 = 5%) to percent values expected by many strategies
        strategy_cls.stop_loss_pct = params.stop_loss_pct * 100.0
        strategy_cls.take_profit_pct = params.take_profit_pct * 100.0
        strategy_cls.position_size_pct = params.position_size_pct * 100.0
        strategy_cls.trailing_stop_pct = params.trailing_stop_pct * 100.0
        strategy_cls.max_consecutive_losses = self.config.max_consecutive_losses
        strategy_cls.max_drawdown_pct = self.config.max_drawdown_pct
        
        return strategy_cls
    
    def generate_parameter_combinations(self) -> List[RiskParameterSet]:
        """Generate all combinations of risk parameters for optimization"""
        combinations = []
        
        for stop_loss, take_profit, position_size, trailing_stop in product(
            self.config.stop_loss_range,
            self.config.take_profit_range,
            self.config.position_size_range,
            self.config.trailing_stop_range
        ):
            # Skip invalid combinations
            if take_profit <= stop_loss:
                continue  # Take profit should be greater than stop loss
            # Ensure trailing stop is not looser than stop loss (allow equal or tighter)
            if trailing_stop > stop_loss:
                continue
            
            combinations.append(RiskParameterSet(
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                position_size_pct=position_size,
                trailing_stop_pct=trailing_stop
            ))
        
        # Optionally downsample to keep compute bounded
        max_combos = int(getattr(self.config, 'max_parameter_combinations', 0) or 0)
        if max_combos > 0 and len(combinations) > max_combos:
            # Evenly sample across the list for coverage
            idx = np.linspace(0, len(combinations) - 1, num=max_combos, dtype=int)
            combinations = [combinations[i] for i in idx]
        return combinations
    
    def calculate_optimization_score(self, train_stats: Dict[str, Any], test_stats: Dict[str, Any]) -> float:
        """
        Calculate optimization score for walkforward selection according to
        self.config.walkforward_selection_criterion: 'oos' | 'is' | 'combined'.
        """
        try:
            # Combine train and test metrics for scoring
            train_return = train_stats.get('Return [%]', 0)
            test_return = test_stats.get('Return [%]', 0)
            train_sharpe = train_stats.get('Sharpe Ratio', 0)
            test_sharpe = test_stats.get('Sharpe Ratio', 0)
            train_drawdown = abs(train_stats.get('Max. Drawdown [%]', 0))
            test_drawdown = abs(test_stats.get('Max. Drawdown [%]', 0))
            
            # Calculate score components
            return_consistency = 1 - abs(train_return - test_return) / (abs(train_return) + 1e-8)
            sharpe_consistency = 1 - abs(train_sharpe - test_sharpe) / (abs(train_sharpe) + 1e-8)
            
            # Penalize high drawdowns
            drawdown_penalty = max(0, (train_drawdown + test_drawdown) / 2 - 20) / 100
            
            criterion = (self.config.walkforward_selection_criterion or "combined").lower()

            if criterion == "oos":
                # Emphasize pure out-of-sample performance
                score = 0.7 * test_return + 0.3 * test_sharpe - drawdown_penalty
            elif criterion == "is":
                # Emphasize in-sample performance only
                score = 0.7 * train_return + 0.3 * train_sharpe - drawdown_penalty
            else:
                # Combined: favor OOS but include consistency
                score = (
                    0.4 * test_return +
                    0.3 * test_sharpe +
                    0.2 * return_consistency +
                    0.1 * sharpe_consistency -
                    drawdown_penalty
                )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating optimization score: {e}")
            return -999  # Very low score for failed calculations
    
    def run_walkforward_optimization(self, data: pd.DataFrame, strategy_cls: type) -> List[OptimizationResult]:
        """Run walkforward optimization for risk parameters"""
        results = []
        n = len(data)

        # Preserve originals and optionally downscale to fit available data while respecting minimum threshold
        orig_train = self.config.walkforward_train_size
        orig_test = self.config.walkforward_test_size
        orig_step = self.config.walkforward_step_size

        try:
            min_total = int(getattr(self.config, 'walkforward_min_total_size', 300) or 300)
            required = self.config.walkforward_train_size + self.config.walkforward_test_size

            if n < min_total:
                self.logger.warning(f"Insufficient data for walkforward optimization: {n} < {min_total}")
                return results

            if n < required:
                # Downscale train/test preserving ~70/30 split, ensure at least small viable windows
                train_size = max(50, int(n * 0.7))
                test_size = max(10, n - train_size)
                step_size = max(1, min(test_size, self.config.walkforward_step_size))
                self.logger.info(
                    f"Downscaling WF windows to fit n={n}: train={train_size}, test={test_size}, step={step_size}"
                )
                self.config.walkforward_train_size = train_size
                self.config.walkforward_test_size = test_size
                self.config.walkforward_step_size = step_size
        
            # Generate parameter combinations
            parameter_combinations = self.generate_parameter_combinations()
            self.logger.info(
                f"Testing {len(parameter_combinations)} parameter combinations: "
                f"SL options={len(self.config.stop_loss_range)}, TP options={len(self.config.take_profit_range)}, "
                f"PS options={len(self.config.position_size_range)}, TS options={len(self.config.trailing_stop_range)}"
            )

            # Run walkforward optimization
            if (self.config.walkforward_window_mode or "rolling").lower() == "expanding":
                # Expanding: train starts at 0 and grows; tests are adjacent windows
                start = 0
                while True:
                    train_end = start + self.config.walkforward_train_size
                    test_end = train_end + self.config.walkforward_test_size
                    if test_end > n:
                        break
                    train_df = data.iloc[:train_end]
                    test_df = data.iloc[train_end:test_end]
                    regime_label = self._classify_regime(test_df)

                    window_results = []
                    for params in parameter_combinations:
                        try:
                            strategy_with_params = self.apply_risk_management_to_strategy(strategy_cls, params)
                            train_stats = self.run_backtest(train_df, strategy_with_params)
                            if train_stats is None:
                                continue
                            test_stats = self.run_backtest(test_df, strategy_with_params)
                            if test_stats is None:
                                continue
                            optimization_score = self.calculate_optimization_score(train_stats, test_stats)
                            result = OptimizationResult(
                                parameter_set=params,
                                train_performance=train_stats,
                                test_performance=test_stats,
                                optimization_score=optimization_score,
                                regime=regime_label
                            )
                            window_results.append(result)
                        except Exception as e:
                            self.logger.warning(f"Error testing parameters {params.to_dict()}: {e}")
                            continue
                    window_results.sort(key=lambda x: x.optimization_score, reverse=True)
                    results.extend(window_results)
                    # grow train by step size
                    self.config.walkforward_train_size += self.config.walkforward_step_size
                    if (self.config.walkforward_train_size + self.config.walkforward_test_size) > n:
                        break
            else:
                # Rolling: slide fixed-size train/test windows
                max_windows = int(getattr(self.config, 'max_walkforward_windows', 0) or 0)
                window_counter = 0
                for start in range(0, n - (self.config.walkforward_train_size + self.config.walkforward_test_size) + 1, self.config.walkforward_step_size):
                    train_end = start + self.config.walkforward_train_size
                    test_end = train_end + self.config.walkforward_test_size

                    if test_end > n:
                        break

                    train_df = data.iloc[start:train_end]
                    test_df = data.iloc[train_end:test_end]
                    regime_label = self._classify_regime(test_df)

                    window_results = []

                    # Test each parameter combination
                    for params in parameter_combinations:
                        try:
                            # Apply parameters to strategy
                            strategy_with_params = self.apply_risk_management_to_strategy(strategy_cls, params)

                            # Run backtest on train period
                            train_stats = self.run_backtest(train_df, strategy_with_params)
                            if train_stats is None:
                                continue

                            # Run backtest on test period
                            test_stats = self.run_backtest(test_df, strategy_with_params)
                            if test_stats is None:
                                continue

                            # Calculate optimization score
                            optimization_score = self.calculate_optimization_score(train_stats, test_stats)

                            # Create optimization result
                            result = OptimizationResult(
                                parameter_set=params,
                                train_performance=train_stats,
                                test_performance=test_stats,
                                optimization_score=optimization_score,
                                regime=regime_label
                            )

                            window_results.append(result)

                        except Exception as e:
                            self.logger.warning(f"Error testing parameters {params.to_dict()}: {e}")
                            continue

                    # Sort results by optimization score
                    window_results.sort(key=lambda x: x.optimization_score, reverse=True)
                    results.extend(window_results)
                    window_counter += 1
                    if max_windows and window_counter >= max_windows:
                        break
            return results
        finally:
            # Restore original configuration
            self.config.walkforward_train_size = orig_train
            self.config.walkforward_test_size = orig_test
            self.config.walkforward_step_size = orig_step

    def _classify_regime(self, df: pd.DataFrame) -> str:
        """Classify a window as bullish, bearish, or sideways based on trend and return."""
        try:
            if len(df) < 5:
                return "unknown"
            total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1.0
            sma_short = df['Close'].rolling(10, min_periods=1).mean().iloc[-1]
            sma_long = df['Close'].rolling(30, min_periods=1).mean().iloc[-1]
            trend = sma_short - sma_long
            if total_return > 0 and trend > 0:
                return "bullish"
            if total_return < 0 and trend < 0:
                return "bearish"
            return "sideways"
        except Exception:
            return "unknown"
    
    def calculate_risk_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return _calc_risk(
            stats,
            max_drawdown_pct=self.config.max_drawdown_pct,
            position_size_pct=self.config.position_size_pct,
            enable_position_sizing=self.config.enable_position_sizing,
        )
    
    def _calculate_risk_management_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall risk management effectiveness score"""
        score = 0.0
        max_score = 5.0
        
        # Drawdown control
        max_drawdown = abs(stats.get('Max. Drawdown [%]', 0))
        if max_drawdown < self.config.max_drawdown_pct:
            score += 1.0
        elif max_drawdown < self.config.max_drawdown_pct * 1.5:
            score += 0.5
        
        # Consecutive loss control
        max_consecutive_losses = stats.get('Max. Consecutive Losses', 0)
        if max_consecutive_losses < self.config.max_consecutive_losses:
            score += 1.0
        elif max_consecutive_losses < self.config.max_consecutive_losses * 1.5:
            score += 0.5
        
        # Sharpe ratio (risk-adjusted returns)
        sharpe = stats.get('Sharpe Ratio', 0)
        if sharpe > 1.5:
            score += 1.0
        elif sharpe > 1.0:
            score += 0.5
        
        # Sortino ratio (downside risk)
        sortino = stats.get('Sortino Ratio', 0)
        if sortino > 2.0:
            score += 1.0
        elif sortino > 1.5:
            score += 0.5
        
        # Calmar ratio (return vs max drawdown)
        calmar = stats.get('Calmar Ratio', 0)
        if calmar > 1.0:
            score += 1.0
        elif calmar > 0.5:
            score += 0.5
        
        return score / max_score
    
    def run_single_backtest(self, strategy_path: str, data_file: str, 
                           strat_result_dir: str, data_name: str) -> Optional[Dict[str, Any]]:
        """Run a single risk-managed backtest and save results"""
        try:
            strategy_cls = self.load_strategy(strategy_path)
            df = self.load_and_validate_data(data_file)
            
            # Extract symbol and timeframe from filename
            symbol, timeframe = self.extract_info_from_filename(data_file)
            
            # Run walkforward optimization if enabled
            optimization_results = []
            if self.config.enable_walkforward_optimization:
                try:
                    optimization_results = self.run_walkforward_optimization(df, strategy_cls)
                    self.logger.info(
                        f"Completed walkforward evaluation with {len(optimization_results)} parameter-window results"
                    )
                except Exception as e:
                    self.logger.error(f"Walkforward optimization failed: {e}")
            
            # Apply default risk management parameters (report-only mode does not auto-optimize)
            strategy_cls = self.apply_risk_management_to_strategy(strategy_cls)
            
            # Run backtest
            stats = self.run_backtest(df, strategy_cls)
            if stats is None:
                return None
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(stats)
            
            # Assess quality
            quality_assessment = self.assess_quality(stats)
            
            # Aggregate per-regime performance from optimization results (report-only)
            regime_agg = {}
            top_params_by_regime = {}
            cross_regime_leaders = []
            if optimization_results:
                df_opt = pd.DataFrame([
                    {
                        'regime': r.regime,
                        'oos_return': r.test_performance.get('Return [%]', 0),
                        'oos_sharpe': r.test_performance.get('Sharpe Ratio', 0),
                        'oos_drawdown': abs(r.test_performance.get('Max. Drawdown [%]', 0)),
                        **r.parameter_set.to_dict()
                    }
                    for r in optimization_results
                ])
                # Persist full parameter eval if desired
                if getattr(self.config, 'save_full_parameter_eval_csv', True):
                    try:
                        csv_out = os.path.join(strat_result_dir, f"{data_name}_walkforward_eval.csv")
                        df_opt.to_csv(csv_out, index=False)
                    except Exception:
                        pass
                # Group by regime for summary
                if not df_opt.empty and 'regime' in df_opt.columns:
                    # Build JSON-safe nested dict (avoid tuple keys from MultiIndex)
                    agg_df = (
                        df_opt.groupby('regime')[['oos_return', 'oos_sharpe', 'oos_drawdown']]
                        .agg(['mean', 'median', 'max', 'min'])
                    )
                    regime_agg = {}
                    for regime_label, row in agg_df.iterrows():
                        regime_key = str(regime_label)
                        regime_agg[regime_key] = {
                            'oos_return': {
                                'mean': float(row[('oos_return', 'mean')]),
                                'median': float(row[('oos_return', 'median')]),
                                'max': float(row[('oos_return', 'max')]),
                                'min': float(row[('oos_return', 'min')]),
                            },
                            'oos_sharpe': {
                                'mean': float(row[('oos_sharpe', 'mean')]),
                                'median': float(row[('oos_sharpe', 'median')]),
                                'max': float(row[('oos_sharpe', 'max')]),
                                'min': float(row[('oos_sharpe', 'min')]),
                            },
                            'oos_drawdown': {
                                'mean': float(row[('oos_drawdown', 'mean')]),
                                'median': float(row[('oos_drawdown', 'median')]),
                                'max': float(row[('oos_drawdown', 'max')]),
                                'min': float(row[('oos_drawdown', 'min')]),
                            },
                        }
                    # Top K parameters per regime
                    metric = 'oos_return' if self.config.report_selection_metric not in ['oos_return', 'oos_sharpe'] else self.config.report_selection_metric
                    for regime, grp in df_opt.groupby('regime'):
                        grp = grp.copy()
                        grp['param_key'] = grp.apply(lambda row: (
                            float(row['stop_loss_pct']), float(row['take_profit_pct']),
                            float(row['position_size_pct']), float(row['trailing_stop_pct'])
                        ), axis=1)
                        agg = grp.groupby('param_key').agg(
                            avg_oos_return=('oos_return', 'mean'),
                            avg_oos_sharpe=('oos_sharpe', 'mean'),
                            avg_oos_drawdown=('oos_drawdown', 'mean'),
                            count=('oos_return', 'count')
                        ).reset_index()
                        agg = agg.sort_values('avg_oos_sharpe' if metric == 'oos_sharpe' else 'avg_oos_return', ascending=False)
                        top_k = agg.head(int(self.config.report_top_k))
                        top_params_by_regime[regime] = [
                            {
                                'parameters': {
                                    'stop_loss_pct': k[0], 'take_profit_pct': k[1],
                                    'position_size_pct': k[2], 'trailing_stop_pct': k[3]
                                },
                                'avg_oos_return': float(row['avg_oos_return']),
                                'avg_oos_sharpe': float(row['avg_oos_sharpe']),
                                'avg_oos_drawdown': float(row['avg_oos_drawdown']),
                                'count': int(row['count'])
                            }
                            for k, row in zip(top_k['param_key'], top_k.to_dict(orient='records'))
                        ]
                    # Cross-regime leaders (overall stability)
                    df_opt['param_key'] = df_opt.apply(lambda row: (
                        float(row['stop_loss_pct']), float(row['take_profit_pct']),
                        float(row['position_size_pct']), float(row['trailing_stop_pct'])
                    ), axis=1)
                    overall = df_opt.groupby('param_key').agg(
                        avg_oos_return=('oos_return', 'mean'),
                        std_oos_return=('oos_return', 'std'),
                        avg_oos_sharpe=('oos_sharpe', 'mean'),
                        max_oos_drawdown=('oos_drawdown', 'max'),
                        regimes_covered=('regime', lambda x: len(set(x)))
                    ).reset_index()
                    overall['stability_score'] = overall['avg_oos_return'] / (overall['std_oos_return'].replace(0, np.nan))
                    overall = overall.sort_values(['regimes_covered', 'stability_score', 'avg_oos_return'], ascending=[False, False, False]).fillna(0)
                    cross_regime_leaders = [
                        {
                            'parameters': {
                                'stop_loss_pct': k[0], 'take_profit_pct': k[1],
                                'position_size_pct': k[2], 'trailing_stop_pct': k[3]
                            },
                            'avg_oos_return': float(r['avg_oos_return']),
                            'std_oos_return': float(0 if np.isnan(r['std_oos_return']) else r['std_oos_return']),
                            'avg_oos_sharpe': float(r['avg_oos_sharpe']),
                            'max_oos_drawdown': float(r['max_oos_drawdown']),
                            'regimes_covered': int(r['regimes_covered'])
                        }
                        for k, r in zip(overall['param_key'].head(int(self.config.report_top_k)), overall.to_dict(orient='records')[: int(self.config.report_top_k)])
                    ]

            # Optional: correlate liquidation intensity with returns on this dataset
            liq_analysis = {}
            try:
                # Dynamically load liquidation fetcher to avoid import issues with hyphenated path
                import importlib.util
                from pathlib import Path as _Path
                liq_path = _Path(__file__).parents[1] / 'tool_box' / 'data-sources' / 'hyperliquid' / 'fetch_liquidations.py'
                if liq_path.exists():
                    spec = importlib.util.spec_from_file_location('hl_liq', str(liq_path))
                    if spec and spec.loader:
                        hl_liq = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(hl_liq)  # type: ignore
                        Fetcher = getattr(hl_liq, 'HyperliquidLiquidationFetcher', None)
                    else:
                        Fetcher = None
                else:
                    Fetcher = None
                if Fetcher is None:
                    raise ImportError('HyperliquidLiquidationFetcher not available')
                fetcher = Fetcher()
                # Fetch a recent small window to avoid heavy calls
                liq_df = fetcher.fetch_liquidations(interval='1h')
                if not liq_df.empty and 'timestamp' in liq_df.columns:
                    liq_df = liq_df.set_index('timestamp').sort_index()
                    # Align to df index frequency using forward fill
                    liq_df_res = liq_df['value'].resample('1H').sum().fillna(0)
                    price_1h = df['Close'].resample('1H').last().ffill()
                    ret = price_1h.pct_change().dropna()
                    aligned = pd.DataFrame({'liq_value': liq_df_res, 'ret': ret}).dropna()
                    if not aligned.empty:
                        corr = float(aligned['liq_value'].corr(aligned['ret']))
                        liq_analysis = {
                            'records': int(len(liq_df)),
                            'ret_corr_with_liq_value': corr,
                            'liq_value_p95': float(aligned['liq_value'].quantile(0.95)),
                        }
            except Exception:
                pass

            # Prepare result dictionary with risk management info (core metrics only)
            result_dict = {
                'strategy': os.path.basename(strategy_path),
                'data_file': data_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'core_metrics': self.select_core_metrics(stats),
                'data_points': len(df),
                'max_consecutive_losses': stats.get('Max. Consecutive Losses', 0),
                'quality_assessment': quality_assessment,
                
                # Risk management parameters
                'risk_management_config': {
                    'stop_loss_pct': self.config.stop_loss_pct,
                    'take_profit_pct': self.config.take_profit_pct,
                    'trailing_stop_pct': self.config.trailing_stop_pct,
                    'position_size_pct': self.config.position_size_pct,
                    'max_consecutive_losses': self.config.max_consecutive_losses,
                    'max_drawdown_pct': self.config.max_drawdown_pct,
                    'enable_position_sizing': self.config.enable_position_sizing,
                    'enable_stop_loss': self.config.enable_stop_loss,
                    'enable_take_profit': self.config.enable_take_profit,
                    'enable_trailing_stop': self.config.enable_trailing_stop,
                    'enable_drawdown_protection': self.config.enable_drawdown_protection,
                    'enable_consecutive_loss_protection': self.config.enable_consecutive_loss_protection
                },
                
                # Risk metrics
                'risk_metrics': risk_metrics,
                
                # Walkforward parameter evaluation (report-only)
                'walkforward_parameter_evaluation': {
                    'enabled': self.config.enable_walkforward_optimization,
                    'total_results': len(optimization_results),
                    'regime_summary': regime_agg,
                    'top_parameters_by_regime': top_params_by_regime,
                    'cross_regime_leaders': cross_regime_leaders,
                    'sample_results': [
                        {
                            'parameters': r.parameter_set.to_dict(),
                            'train_core': self.select_core_metrics(r.train_performance),
                            'test_core': self.select_core_metrics(r.test_performance),
                            'optimization_score': float(r.optimization_score),
                            'regime': getattr(r, 'regime', 'unknown'),
                            'window_index': getattr(r, 'window_index', -1),
                        }
                        for r in optimization_results[:100]
                    ]
                },
                'liquidation_analysis': liq_analysis
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
        """Main function to run risk-managed backtests with walkforward optimization"""
        # Discover strategies
        strategies = self.discover_strategies()
        self.logger.info(f"Starting risk-managed backtesting with walkforward optimization")
        try:
            total_data_files = len(self.discover_data_files())
        except Exception:
            total_data_files = 0
        self.logger.info(f"Discovered {len(strategies)} strategies and {total_data_files} data files (overall).")
        
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
            
            # Create risk management summary
            df = pd.DataFrame(all_results)
            risk_summary = {
                'total_backtests': len(all_results),
                'avg_risk_management_score': df['risk_metrics'].apply(
                    lambda x: x.get('risk_management_score', 0) if isinstance(x, dict) else 0
                ).mean(),
                'avg_drawdown': df['core_metrics'].apply(
                    lambda x: x.get('Max. Drawdown [%]', 0) if isinstance(x, dict) else 0
                ).mean(),
                'avg_consecutive_losses': df['max_consecutive_losses'].mean(),
                'optimization_enabled': self.config.enable_walkforward_optimization,
                'risk_management_config': {
                    'stop_loss_pct': self.config.stop_loss_pct,
                    'take_profit_pct': self.config.take_profit_pct,
                    'trailing_stop_pct': self.config.trailing_stop_pct,
                    'position_size_pct': self.config.position_size_pct,
                    'max_consecutive_losses': self.config.max_consecutive_losses,
                    'max_drawdown_pct': self.config.max_drawdown_pct
                },
                'resume_info': self.get_resume_info(results_dir)
            }
            
            risk_summary_path = os.path.join(results_dir, 'risk_management_summary')
            self.save_results(risk_summary, risk_summary_path)
            
            self.logger.info(f"Saved {len(all_results)} total results to {results_dir}")
        else:
            self.logger.warning("No results to save")
        
        if self.shutdown_requested:
            self.logger.info("Backtest run completed with early termination due to shutdown request.")
        else:
            self.logger.info("Risk-managed backtest run completed successfully.")
        
        return all_results 