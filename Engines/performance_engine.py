#!/usr/bin/env python3
"""
Performance Analytics Engine

A sophisticated engine that provides:
- Advanced performance metrics calculation
- Performance attribution and decomposition
- Benchmark comparison and analysis
- Performance visualization and reporting
- Performance forecasting and prediction
- Performance optimization recommendations
- Real-time performance monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetric(Enum):
    """Performance metric enumeration"""
    RETURN = "return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VAR = "var"
    CVAR = "cvar"
    BETA = "beta"
    ALPHA = "alpha"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    JENSEN_ALPHA = "jensen_alpha"

class BenchmarkType(Enum):
    """Benchmark type enumeration"""
    MARKET_INDEX = "market_index"
    RISK_FREE = "risk_free"
    CUSTOM = "custom"
    PEER_GROUP = "peer_group"

@dataclass
class PerformanceEngineConfig:
    """Configuration for the performance engine"""
    # Paths
    data_path: str = "./fetched_data"
    results_path: str = "./Results"
    
    # Performance Analysis
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_symbol: str = "SPY"  # S&P 500 ETF as default benchmark
    
    # Metrics Configuration
    primary_metrics: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.RETURN,
        PerformanceMetric.VOLATILITY,
        PerformanceMetric.SHARPE_RATIO,
        PerformanceMetric.MAX_DRAWDOWN
    ])
    
    advanced_metrics: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.SORTINO_RATIO,
        PerformanceMetric.CALMAR_RATIO,
        PerformanceMetric.VAR,
        PerformanceMetric.CVAR,
        PerformanceMetric.BETA,
        PerformanceMetric.ALPHA,
        PerformanceMetric.INFORMATION_RATIO
    ])
    
    # Time Periods
    analysis_periods: List[str] = field(default_factory=lambda: [
        "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "ALL"
    ])
    
    # Rolling Analysis
    rolling_windows: List[int] = field(default_factory=lambda: [30, 60, 90, 252])
    
    # VaR Configuration
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_methods: List[str] = field(default_factory=lambda: ["historical", "parametric", "monte_carlo"])
    
    # Performance Attribution
    enable_attribution: bool = True
    attribution_factors: List[str] = field(default_factory=lambda: [
        "sector", "size", "value", "momentum", "quality"
    ])
    
    # Visualization
    enable_plots: bool = True
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    plot_dpi: int = 300
    
    # Forecasting
    enable_forecasting: bool = True
    forecast_horizons: List[int] = field(default_factory=lambda: [30, 60, 90, 252])
    forecast_methods: List[str] = field(default_factory=lambda: ["arima", "garch", "monte_carlo"])
    
    # Performance
    max_workers: int = 4
    chunk_size: int = 1000
    
    # Output
    save_performance_reports: bool = True
    save_attribution_analysis: bool = True
    save_forecasts: bool = True
    save_visualizations: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic Metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    
    # Risk Metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Benchmark Comparison
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    
    # Additional Metrics
    win_rate: float
    profit_factor: float
    recovery_factor: float
    sterling_ratio: float
    burke_ratio: float
    kappa_3: float
    
    # Time Period
    start_date: datetime
    end_date: datetime
    num_periods: int
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAttribution:
    """Performance attribution analysis"""
    total_attribution: float
    factor_attributions: Dict[str, float]
    security_attributions: Dict[str, float]
    interaction_effects: Dict[str, float]
    residual: float
    timestamp: datetime = field(default_factory=datetime.now)

class PerformanceCalculator:
    """Advanced performance metrics calculator"""
    
    def __init__(self, config: PerformanceEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_comprehensive_metrics(self, returns: pd.Series, 
                                      benchmark_returns: pd.Series = None,
                                      risk_free_rate: float = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config.risk_free_rate
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Drawdown metrics
            equity_curve = (1 + returns).cumprod()
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min()
            current_drawdown = drawdown.iloc[-1]
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
            
            # Benchmark comparison metrics
            beta = 0.0
            alpha = 0.0
            tracking_error = 0.0
            information_ratio = 0.0
            treynor_ratio = 0.0
            jensen_alpha = 0.0
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align returns
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                
                if len(aligned_returns) > 1:
                    # Beta
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                    
                    # Alpha
                    benchmark_return = aligned_benchmark.mean() * 252
                    alpha = annualized_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
                    
                    # Tracking error
                    tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
                    
                    # Information ratio
                    excess_return = (aligned_returns - aligned_benchmark).mean() * 252
                    information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
                    
                    # Treynor ratio
                    treynor_ratio = excess_returns.mean() * 252 / beta if beta != 0 else 0
                    
                    # Jensen's alpha
                    jensen_alpha = alpha
            
            # Additional metrics
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Recovery factor
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sterling ratio
            avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
            sterling_ratio = annualized_return / abs(avg_drawdown) if avg_drawdown != 0 else 0
            
            # Burke ratio
            drawdown_squared = (drawdown[drawdown < 0] ** 2).sum()
            burke_ratio = annualized_return / np.sqrt(drawdown_squared) if drawdown_squared > 0 else 0
            
            # Kappa 3 (third moment)
            kappa_3 = self._calculate_kappa(returns, 3)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio,
                jensen_alpha=jensen_alpha,
                win_rate=win_rate,
                profit_factor=profit_factor,
                recovery_factor=recovery_factor,
                sterling_ratio=sterling_ratio,
                burke_ratio=burke_ratio,
                kappa_3=kappa_3,
                start_date=returns.index[0],
                end_date=returns.index[-1],
                num_periods=len(returns)
            )
        
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def _calculate_kappa(self, returns: pd.Series, order: int) -> float:
        """Calculate Kappa ratio of specified order"""
        try:
            if order < 2:
                return 0.0
            
            # Calculate threshold return (target return)
            threshold_return = 0.0  # Can be adjusted
            
            # Calculate excess returns
            excess_returns = returns - threshold_return
            
            # Calculate lower partial moment
            negative_excess = excess_returns[excess_returns < 0]
            if len(negative_excess) == 0:
                return float('inf')
            
            lower_partial_moment = (negative_excess ** order).mean()
            
            if lower_partial_moment == 0:
                return float('inf')
            
            # Calculate Kappa ratio
            kappa = excess_returns.mean() / (lower_partial_moment ** (1/order))
            
            return kappa
        
        except Exception as e:
            return 0.0
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                window: int) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        try:
            rolling_metrics = pd.DataFrame(index=returns.index)
            
            # Rolling returns
            rolling_metrics['rolling_return'] = returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Rolling volatility
            rolling_metrics['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio
            excess_returns = returns - self.config.risk_free_rate / 252
            rolling_metrics['rolling_sharpe'] = (
                excess_returns.rolling(window=window).mean() / 
                returns.rolling(window=window).std() * np.sqrt(252)
            )
            
            # Rolling drawdown
            equity_curve = (1 + returns).cumprod()
            rolling_max = equity_curve.rolling(window=window).max()
            rolling_metrics['rolling_drawdown'] = (equity_curve - rolling_max) / rolling_max
            
            # Rolling VaR
            rolling_metrics['rolling_var_95'] = returns.rolling(window=window).apply(
                lambda x: np.percentile(x, 5)
            )
            
            return rolling_metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()
    
    def calculate_period_metrics(self, returns: pd.Series, 
                               periods: List[str]) -> Dict[str, PerformanceMetrics]:
        """Calculate performance metrics for different time periods"""
        try:
            period_metrics = {}
            
            for period in periods:
                if period == "ALL":
                    period_returns = returns
                else:
                    # Calculate period end date
                    end_date = returns.index[-1]
                    
                    if period == "1M":
                        start_date = end_date - timedelta(days=30)
                    elif period == "3M":
                        start_date = end_date - timedelta(days=90)
                    elif period == "6M":
                        start_date = end_date - timedelta(days=180)
                    elif period == "1Y":
                        start_date = end_date - timedelta(days=365)
                    elif period == "2Y":
                        start_date = end_date - timedelta(days=730)
                    elif period == "3Y":
                        start_date = end_date - timedelta(days=1095)
                    elif period == "5Y":
                        start_date = end_date - timedelta(days=1825)
                    else:
                        continue
                    
                    period_returns = returns[returns.index >= start_date]
                
                if len(period_returns) < 30:  # Need minimum data
                    continue
                
                # Calculate metrics for this period
                metrics = self.calculate_comprehensive_metrics(period_returns)
                period_metrics[period] = metrics
            
            return period_metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating period metrics: {e}")
            return {}

class PerformanceAttributor:
    """Performance attribution analysis engine"""
    
    def __init__(self, config: PerformanceEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_attribution(self, portfolio_returns: pd.Series, 
                            factor_returns: Dict[str, pd.Series],
                            factor_exposures: Dict[str, float]) -> PerformanceAttribution:
        """Calculate performance attribution"""
        try:
            # Align all data
            aligned_data = pd.DataFrame({'portfolio': portfolio_returns})
            
            for factor, returns in factor_returns.items():
                aligned_data[factor] = returns
            
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 30:
                raise ValueError("Insufficient data for attribution analysis")
            
            # Calculate factor contributions
            factor_attributions = {}
            total_factor_contribution = 0.0
            
            for factor, exposure in factor_exposures.items():
                if factor in aligned_data.columns:
                    factor_contribution = exposure * aligned_data[factor].mean() * 252
                    factor_attributions[factor] = factor_contribution
                    total_factor_contribution += factor_contribution
            
            # Calculate total attribution
            portfolio_return = aligned_data['portfolio'].mean() * 252
            total_attribution = portfolio_return
            
            # Calculate residual (unexplained return)
            residual = total_attribution - total_factor_contribution
            
            # Security attributions (simplified - would need individual security data)
            security_attributions = {}
            
            # Interaction effects (simplified)
            interaction_effects = {}
            
            return PerformanceAttribution(
                total_attribution=total_attribution,
                factor_attributions=factor_attributions,
                security_attributions=security_attributions,
                interaction_effects=interaction_effects,
                residual=residual
            )
        
        except Exception as e:
            self.logger.error(f"Error calculating attribution: {e}")
            return PerformanceAttribution(
                total_attribution=0.0,
                factor_attributions={},
                security_attributions={},
                interaction_effects={},
                residual=0.0
            )

class PerformanceForecaster:
    """Performance forecasting engine"""
    
    def __init__(self, config: PerformanceEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forecast_performance(self, returns: pd.Series, 
                           horizons: List[int]) -> Dict[int, Dict[str, float]]:
        """Forecast performance for different horizons"""
        try:
            forecasts = {}
            
            for horizon in horizons:
                if horizon > len(returns):
                    continue
                
                # Simple forecasting methods
                forecast_metrics = {}
                
                # Historical mean forecast
                historical_mean = returns.mean()
                forecast_metrics['mean_return'] = historical_mean * horizon
                
                # Historical volatility forecast
                historical_vol = returns.std()
                forecast_metrics['volatility'] = historical_vol * np.sqrt(horizon)
                
                # VaR forecast
                forecast_metrics['var_95'] = np.percentile(returns, 5) * np.sqrt(horizon)
                forecast_metrics['var_99'] = np.percentile(returns, 1) * np.sqrt(horizon)
                
                # Sharpe ratio forecast (simplified)
                risk_free_rate = self.config.risk_free_rate
                excess_return = historical_mean - risk_free_rate / 252
                forecast_metrics['sharpe_ratio'] = excess_return / historical_vol * np.sqrt(252) if historical_vol > 0 else 0
                
                # Confidence intervals
                forecast_metrics['return_ci_lower'] = forecast_metrics['mean_return'] - 1.96 * forecast_metrics['volatility']
                forecast_metrics['return_ci_upper'] = forecast_metrics['mean_return'] + 1.96 * forecast_metrics['volatility']
                
                forecasts[horizon] = forecast_metrics
            
            return forecasts
        
        except Exception as e:
            self.logger.error(f"Error forecasting performance: {e}")
            return {}

class PerformanceVisualizer:
    """Performance visualization engine"""
    
    def __init__(self, config: PerformanceEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_performance_plots(self, returns: pd.Series, 
                               metrics: PerformanceMetrics,
                               output_dir: str):
        """Create comprehensive performance visualization plots"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Equity curve and drawdown
            self._plot_equity_curve(returns, output_path)
            
            # 2. Rolling metrics
            self._plot_rolling_metrics(returns, output_path)
            
            # 3. Return distribution
            self._plot_return_distribution(returns, output_path)
            
            # 4. Performance metrics summary
            self._plot_metrics_summary(metrics, output_path)
            
            # 5. Risk-return scatter
            self._plot_risk_return(returns, output_path)
            
            self.logger.info(f"Created performance plots in {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error creating performance plots: {e}")
    
    def _plot_equity_curve(self, returns: pd.Series, output_path: Path):
        """Plot equity curve and drawdown"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Equity curve
            equity_curve = (1 + returns).cumprod()
            ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue')
            ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax2.plot(drawdown.index, drawdown.values, linewidth=1, color='red')
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            for fmt in self.config.plot_formats:
                plt.savefig(output_path / f'equity_curve.{fmt}', 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
    
    def _plot_rolling_metrics(self, returns: pd.Series, output_path: Path):
        """Plot rolling performance metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Rolling returns
            rolling_returns = returns.rolling(window=30).apply(lambda x: (1 + x).prod() - 1)
            axes[0, 0].plot(rolling_returns.index, rolling_returns.values * 100, linewidth=1)
            axes[0, 0].set_title('30-Day Rolling Returns', fontweight='bold')
            axes[0, 0].set_ylabel('Return (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
            axes[0, 1].plot(rolling_vol.index, rolling_vol.values, linewidth=1, color='orange')
            axes[0, 1].set_title('30-Day Rolling Volatility', fontweight='bold')
            axes[0, 1].set_ylabel('Volatility (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Rolling Sharpe ratio
            excess_returns = returns - self.config.risk_free_rate / 252
            rolling_sharpe = (excess_returns.rolling(window=30).mean() / 
                            returns.rolling(window=30).std() * np.sqrt(252))
            axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1, color='green')
            axes[1, 0].set_title('30-Day Rolling Sharpe Ratio', fontweight='bold')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Rolling drawdown
            equity_curve = (1 + returns).cumprod()
            rolling_max = equity_curve.rolling(window=30).max()
            rolling_dd = (equity_curve - rolling_max) / rolling_max * 100
            axes[1, 1].fill_between(rolling_dd.index, rolling_dd.values, 0, alpha=0.3, color='red')
            axes[1, 1].set_title('30-Day Rolling Drawdown', fontweight='bold')
            axes[1, 1].set_ylabel('Drawdown (%)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            for fmt in self.config.plot_formats:
                plt.savefig(output_path / f'rolling_metrics.{fmt}', 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error plotting rolling metrics: {e}")
    
    def _plot_return_distribution(self, returns: pd.Series, output_path: Path):
        """Plot return distribution"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(returns * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(returns.mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
            ax1.set_title('Return Distribution', fontweight='bold')
            ax1.set_xlabel('Daily Return (%)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            for fmt in self.config.plot_formats:
                plt.savefig(output_path / f'return_distribution.{fmt}', 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error plotting return distribution: {e}")
    
    def _plot_metrics_summary(self, metrics: PerformanceMetrics, output_path: Path):
        """Plot performance metrics summary"""
        try:
            # Create metrics summary
            metrics_data = {
                'Total Return': f"{metrics.total_return:.2%}",
                'Annualized Return': f"{metrics.annualized_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Sortino Ratio': f"{metrics.sortino_ratio:.2f}",
                'Calmar Ratio': f"{metrics.calmar_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'VaR (95%)': f"{metrics.var_95:.2%}",
                'CVaR (95%)': f"{metrics.cvar_95:.2%}",
                'Beta': f"{metrics.beta:.2f}",
                'Alpha': f"{metrics.alpha:.2%}",
                'Information Ratio': f"{metrics.information_ratio:.2f}"
            }
            
            # Create text-based summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, 'Performance Metrics Summary', 
                   ha='center', va='top', fontsize=16, fontweight='bold')
            
            # Metrics
            y_pos = 0.85
            for metric, value in metrics_data.items():
                ax.text(0.1, y_pos, f"{metric}:", fontsize=12, fontweight='bold')
                ax.text(0.6, y_pos, value, fontsize=12)
                y_pos -= 0.05
            
            plt.tight_layout()
            
            for fmt in self.config.plot_formats:
                plt.savefig(output_path / f'metrics_summary.{fmt}', 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error plotting metrics summary: {e}")
    
    def _plot_risk_return(self, returns: pd.Series, output_path: Path):
        """Plot risk-return scatter"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate risk and return
            portfolio_return = returns.mean() * 252
            portfolio_risk = returns.std() * np.sqrt(252)
            
            # Plot portfolio
            ax.scatter(portfolio_risk * 100, portfolio_return * 100, 
                      s=200, color='red', alpha=0.7, label='Portfolio')
            
            # Add risk-free rate
            risk_free_rate = self.config.risk_free_rate * 100
            ax.axhline(y=risk_free_rate, color='green', linestyle='--', 
                      alpha=0.7, label=f'Risk-Free Rate ({risk_free_rate:.1f}%)')
            
            # Add efficient frontier (simplified)
            risk_range = np.linspace(0, portfolio_risk * 2, 100)
            efficient_frontier = risk_free_rate + (portfolio_return - risk_free_rate) * (risk_range / portfolio_risk)
            ax.plot(risk_range * 100, efficient_frontier * 100, 
                   color='blue', linestyle='-', alpha=0.5, label='Efficient Frontier')
            
            ax.set_xlabel('Risk (Volatility %)', fontsize=12)
            ax.set_ylabel('Return (%)', fontsize=12)
            ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            for fmt in self.config.plot_formats:
                plt.savefig(output_path / f'risk_return.{fmt}', 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error plotting risk-return: {e}")

class PerformanceEngine:
    """Performance analytics engine"""
    
    def __init__(self, config: PerformanceEngineConfig = None):
        self.config = config or PerformanceEngineConfig()
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.performance_calculator = PerformanceCalculator(self.config)
        self.performance_attributor = PerformanceAttributor(self.config)
        self.performance_forecaster = PerformanceForecaster(self.config)
        self.performance_visualizer = PerformanceVisualizer(self.config)
        
        self.logger.info("Performance Engine initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if requested
        if self.config.log_to_file:
            log_file = f"performance_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            logging.getLogger().addHandler(file_handler)
    
    def analyze_performance(self, data: pd.DataFrame, 
                          benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        try:
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate benchmark returns if provided
            benchmark_returns = None
            if benchmark_data is not None:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            # Calculate comprehensive metrics
            metrics = self.performance_calculator.calculate_comprehensive_metrics(
                returns, benchmark_returns
            )
            
            # Calculate period metrics
            period_metrics = self.performance_calculator.calculate_period_metrics(
                returns, self.config.analysis_periods
            )
            
            # Calculate rolling metrics
            rolling_metrics = {}
            for window in self.config.rolling_windows:
                rolling_metrics[f'window_{window}'] = self.performance_calculator.calculate_rolling_metrics(
                    returns, window
                )
            
            # Performance attribution (simplified)
            attribution = None
            if self.config.enable_attribution:
                # This would require factor data - simplified for now
                attribution = PerformanceAttribution(
                    total_attribution=metrics.annualized_return,
                    factor_attributions={},
                    security_attributions={},
                    interaction_effects={},
                    residual=metrics.annualized_return
                )
            
            # Performance forecasting
            forecasts = None
            if self.config.enable_forecasting:
                forecasts = self.performance_forecaster.forecast_performance(
                    returns, self.config.forecast_horizons
                )
            
            return {
                'metrics': metrics,
                'period_metrics': period_metrics,
                'rolling_metrics': rolling_metrics,
                'attribution': attribution,
                'forecasts': forecasts,
                'returns': returns,
                'benchmark_returns': benchmark_returns
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {}
    
    async def run(self) -> List[Dict[str, Any]]:
        """Run the performance engine"""
        try:
            self.logger.info("Starting Performance Engine")
            start_time = time.time()
            
            # Discover data files
            data_path = Path(self.config.data_path)
            data_files = list(data_path.rglob("*.csv"))
            
            if not data_files:
                self.logger.warning("No data files found")
                return []
            
            all_results = []
            
            for data_file in data_files:
                try:
                    self.logger.info(f"Processing {data_file.name}")
                    
                    # Load data
                    data = pd.read_csv(data_file, parse_dates=True, index_col=0)
                    
                    if len(data) < 252:  # Need at least 1 year of data
                        self.logger.warning(f"Insufficient data in {data_file.name}: {len(data)}")
                        continue
                    
                    # Analyze performance
                    analysis = self.analyze_performance(data)
                    
                    if not analysis:
                        continue
                    
                    # Add metadata
                    analysis['data_file'] = data_file.stem
                    analysis['analysis_timestamp'] = datetime.now().isoformat()
                    
                    # Create visualizations
                    if self.config.enable_plots and 'metrics' in analysis:
                        viz_dir = Path(self.config.results_path) / "visualizations" / data_file.stem
                        self.performance_visualizer.create_performance_plots(
                            analysis['returns'], analysis['metrics'], str(viz_dir)
                        )
                    
                    all_results.append(analysis)
                    
                    self.logger.info(f"Completed performance analysis for {data_file.name}")
                    self.logger.info(f"Annualized Return: {analysis['metrics'].annualized_return:.2%}")
                    self.logger.info(f"Sharpe Ratio: {analysis['metrics'].sharpe_ratio:.2f}")
                    self.logger.info(f"Max Drawdown: {analysis['metrics'].max_drawdown:.2%}")
                
                except Exception as e:
                    self.logger.error(f"Error processing {data_file.name}: {e}")
                    continue
            
            # Save results
            if all_results:
                await self._save_results(all_results)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Performance Engine completed in {execution_time:.2f} seconds")
            self.logger.info(f"Analyzed {len(all_results)} datasets")
            
            return all_results
        
        except Exception as e:
            self.logger.error(f"Error in Performance Engine: {e}")
            return []
    
    async def _save_results(self, results: List[Dict[str, Any]]):
        """Save performance analysis results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(self.config.results_path) / f"performance_engine_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert results to dict
            results_dict = []
            for result in results:
                result_dict = {
                    'data_file': result['data_file'],
                    'analysis_timestamp': result['analysis_timestamp'],
                    'metrics': {
                        'total_return': result['metrics'].total_return,
                        'annualized_return': result['metrics'].annualized_return,
                        'volatility': result['metrics'].volatility,
                        'sharpe_ratio': result['metrics'].sharpe_ratio,
                        'sortino_ratio': result['metrics'].sortino_ratio,
                        'calmar_ratio': result['metrics'].calmar_ratio,
                        'max_drawdown': result['metrics'].max_drawdown,
                        'current_drawdown': result['metrics'].current_drawdown,
                        'var_95': result['metrics'].var_95,
                        'var_99': result['metrics'].var_99,
                        'cvar_95': result['metrics'].cvar_95,
                        'cvar_99': result['metrics'].cvar_99,
                        'beta': result['metrics'].beta,
                        'alpha': result['metrics'].alpha,
                        'tracking_error': result['metrics'].tracking_error,
                        'information_ratio': result['metrics'].information_ratio,
                        'treynor_ratio': result['metrics'].treynor_ratio,
                        'jensen_alpha': result['metrics'].jensen_alpha,
                        'win_rate': result['metrics'].win_rate,
                        'profit_factor': result['metrics'].profit_factor,
                        'recovery_factor': result['metrics'].recovery_factor,
                        'sterling_ratio': result['metrics'].sterling_ratio,
                        'burke_ratio': result['metrics'].burke_ratio,
                        'kappa_3': result['metrics'].kappa_3,
                        'start_date': result['metrics'].start_date.isoformat(),
                        'end_date': result['metrics'].end_date.isoformat(),
                        'num_periods': result['metrics'].num_periods
                    }
                }
                
                # Add period metrics
                if 'period_metrics' in result:
                    period_metrics_dict = {}
                    for period, metrics in result['period_metrics'].items():
                        period_metrics_dict[period] = {
                            'total_return': metrics.total_return,
                            'annualized_return': metrics.annualized_return,
                            'volatility': metrics.volatility,
                            'sharpe_ratio': metrics.sharpe_ratio,
                            'max_drawdown': metrics.max_drawdown
                        }
                    result_dict['period_metrics'] = period_metrics_dict
                
                # Add forecasts
                if 'forecasts' in result and result['forecasts']:
                    result_dict['forecasts'] = result['forecasts']
                
                results_dict.append(result_dict)
            
            # Save JSON
            json_file = results_dir / "performance_analysis.json"
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # Save summary
            summary_file = results_dir / "performance_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Performance Engine Analysis Summary\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"Total Analyses: {len(results)}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                f.write("Performance Summary:\n")
                f.write("-" * 20 + "\n")
                for result in results:
                    f.write(f"{result['data_file']}:\n")
                    f.write(f"  Annualized Return: {result['metrics'].annualized_return:.2%}\n")
                    f.write(f"  Volatility: {result['metrics'].volatility:.2%}\n")
                    f.write(f"  Sharpe Ratio: {result['metrics'].sharpe_ratio:.2f}\n")
                    f.write(f"  Max Drawdown: {result['metrics'].max_drawdown:.2%}\n")
                    f.write(f"  Beta: {result['metrics'].beta:.2f}\n")
                    f.write(f"  Alpha: {result['metrics'].alpha:.2%}\n\n")
            
            self.logger.info(f"Saved results to {results_dir}")
        
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

async def main():
    """Main function to run the performance engine"""
    config = PerformanceEngineConfig(
        enable_plots=True,
        enable_attribution=True,
        enable_forecasting=True,
        save_performance_reports=True,
        save_visualizations=True
    )
    
    engine = PerformanceEngine(config)
    results = await engine.run()
    
    print(f"\nPerformance Engine Results:")
    print(f"Total Analyses: {len(results)}")
    
    if results:
        avg_return = np.mean([r['metrics'].annualized_return for r in results])
        avg_sharpe = np.mean([r['metrics'].sharpe_ratio for r in results])
        avg_drawdown = np.mean([r['metrics'].max_drawdown for r in results])
        
        print(f"Average Annualized Return: {avg_return:.2%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Max Drawdown: {avg_drawdown:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
