"""
Abstract Base Engine for Backtesting

This module provides the abstract base class for all backtesting engines.
All engines must implement the methods defined in this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .base_strategy import BaseStrategy
from .base_data_handler import BaseDataHandler
from .base_risk_manager import BaseRiskManager


@dataclass
class EngineConfig:
    """Base configuration for all engines"""
    
    # Paths
    data_path: str = "./data"
    strategies_path: str = "./strategies"
    results_path: str = "./results"
    
    # Backtest parameters
    initial_cash: float = 100000.0
    commission: float = 0.002
    slippage: float = 0.0001
    
    # Performance settings
    enable_optimization: bool = False
    optimization_method: str = "grid_search"  # grid_search, genetic, bayesian
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, calmar_ratio
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_drawdown: float = 0.2  # 20% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    
    # Output options
    save_results: bool = True
    save_plots: bool = True
    verbose: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "backtest.log"


@dataclass
class BacktestResult:
    """Standard result structure for all backtests"""
    
    # Basic info
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    volatility: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk
    
    # Additional data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BaseEngine(ABC):
    """
    Abstract base class for all backtesting engines.
    
    This class defines the interface that all backtesting engines must implement.
    It provides common functionality and enforces consistent behavior across
    different engine implementations.
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize the base engine.
        
        Args:
            config: Engine configuration object
        """
        self.config = config
        self.logger = self._setup_logging()
        self.data_handler: Optional[BaseDataHandler] = None
        self.risk_manager: Optional[BaseRiskManager] = None
        self.strategies: List[BaseStrategy] = []
        self.results: List[BacktestResult] = []
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the engine"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if self.config.log_to_file:
            handler = logging.FileHandler(self.config.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_config(self):
        """Validate engine configuration"""
        if not Path(self.config.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.config.data_path}")
        
        if not Path(self.config.strategies_path).exists():
            raise ValueError(f"Strategies path does not exist: {self.config.strategies_path}")
        
        if self.config.initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        
        if self.config.commission < 0:
            raise ValueError("Commission cannot be negative")
    
    @abstractmethod
    def load_data(self, symbol: str, timeframe: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load market data for backtesting.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            timeframe: Data timeframe (e.g., '1h', '1d')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def load_strategy(self, strategy_name: str, **parameters) -> BaseStrategy:
        """
        Load a trading strategy.
        
        Args:
            strategy_name: Name of the strategy to load
            **parameters: Strategy parameters
            
        Returns:
            Loaded strategy instance
        """
        pass
    
    @abstractmethod
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                    **kwargs) -> BacktestResult:
        """
        Run a backtest for a single strategy.
        
        Args:
            strategy: Strategy to backtest
            data: Market data for backtesting
            **kwargs: Additional backtest parameters
            
        Returns:
            Backtest result object
        """
        pass
    
    @abstractmethod
    def run_portfolio_backtest(self, strategies: List[BaseStrategy], 
                             data_dict: Dict[str, pd.DataFrame], **kwargs) -> List[BacktestResult]:
        """
        Run a portfolio backtest with multiple strategies.
        
        Args:
            strategies: List of strategies to backtest
            data_dict: Dictionary mapping symbols to market data
            **kwargs: Additional backtest parameters
            
        Returns:
            List of backtest results
        """
        pass
    
    @abstractmethod
    def optimize_strategy(self, strategy: BaseStrategy, data: pd.DataFrame, 
                         param_ranges: Dict[str, List[Any]], **kwargs) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy: Strategy to optimize
            data: Market data for optimization
            param_ranges: Parameter ranges to test
            **kwargs: Additional optimization parameters
            
        Returns:
            Tuple of (best_parameters, best_result)
        """
        pass
    
    def save_results(self, results: Union[BacktestResult, List[BacktestResult]], 
                    output_path: Optional[str] = None):
        """
        Save backtest results to disk.
        
        Args:
            results: Single result or list of results to save
            output_path: Custom output path (optional)
        """
        if not self.config.save_results:
            return
        
        if output_path is None:
            output_path = self.config.results_path
        
        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, BacktestResult):
            results = [results]
        
        for result in results:
            # Save detailed results
            result_file = Path(output_path) / f"{result.strategy_name}_{result.symbol}_{result.timeframe}.json"
            self._save_result_json(result, result_file)
            
            # Save equity curve
            if not result.equity_curve.empty:
                equity_file = Path(output_path) / f"{result.strategy_name}_{result.symbol}_{result.timeframe}_equity.csv"
                result.equity_curve.to_csv(equity_file)
    
    def _save_result_json(self, result: BacktestResult, file_path: Path):
        """Save a single result to JSON format"""
        import json
        
        # Convert datetime objects to strings for JSON serialization
        result_dict = {
            'strategy_name': result.strategy_name,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'largest_win': result.largest_win,
            'largest_loss': result.largest_loss,
            'volatility': result.volatility,
            'var_95': result.var_95,
            'cvar_95': result.cvar_95,
            'parameters': result.parameters,
            'execution_time': result.execution_time,
            'timestamp': result.timestamp.isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all backtest results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        returns = [r.total_return for r in self.results]
        sharpe_ratios = [r.sharpe_ratio for r in self.results]
        max_drawdowns = [r.max_drawdown for r in self.results]
        win_rates = [r.win_rate for r in self.results]
        
        return {
            'total_backtests': len(self.results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'best_strategy': max(self.results, key=lambda x: x.total_return).strategy_name,
            'worst_strategy': min(self.results, key=lambda x: x.total_return).strategy_name
        }
    
    def cleanup(self):
        """Clean up resources and close connections"""
        self.logger.info("Cleaning up engine resources")
        # Override in subclasses if needed
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
