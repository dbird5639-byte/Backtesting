"""
Abstract Base Strategy for Trading

This module provides the abstract base class for all trading strategies.
All strategies must implement the methods defined in this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .base_risk_manager import BaseRiskManager


@dataclass
class StrategyConfig:
    """Base configuration for all strategies"""
    
    # Strategy identification
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Trading parameters
    symbol: str = ""
    timeframe: str = "1h"
    
    # Risk parameters
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.02  # 2% trailing stop
    
    # Strategy-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_window: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "strategy.log"


@dataclass
class Signal:
    """Trading signal structure"""
    
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    symbol: str
    price: float
    quantity: float
    confidence: float = 1.0  # Signal confidence (0.0 to 1.0)
    
    # Additional signal metadata
    reason: str = ""
    indicators: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Trade execution structure"""
    
    # Trade identification
    trade_id: str
    strategy_name: str
    symbol: str
    
    # Trade details
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    side: str  # 'long' or 'short'
    
    # Financial details
    entry_value: float
    exit_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Status
    status: str = "open"  # 'open', 'closed', 'cancelled'
    close_reason: str = ""  # 'stop_loss', 'take_profit', 'signal', 'manual'
    
    # Metadata
    entry_signal: Optional[Signal] = None
    exit_signal: Optional[Signal] = None
    notes: str = ""


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    It provides common functionality and enforces consistent behavior across
    different strategy implementations.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the base strategy.
        
        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.logger = self._setup_logging()
        self.risk_manager: Optional[BaseRiskManager] = None
        
        # Performance tracking
        self.signals: List[Signal] = []
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.drawdown_curve: pd.Series = pd.Series(dtype=float)
        
        # State tracking
        self.current_position: Optional[Trade] = None
        self.is_initialized: bool = False
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the strategy"""
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
        """Validate strategy configuration"""
        if not self.config.name:
            raise ValueError("Strategy name is required")
        
        if self.config.max_position_size <= 0 or self.config.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        
        if self.config.stop_loss_pct < 0:
            raise ValueError("Stop loss percentage cannot be negative")
    
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> bool:
        """
        Initialize the strategy with historical data.
        
        This method is called once at the beginning of backtesting
        to set up any necessary indicators or state variables.
        
        Args:
            data: Historical market data for initialization
            
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        This is the core method that all strategies must implement.
        It analyzes the market data and returns a list of trading signals.
        
        Args:
            data: Current market data
            
        Returns:
            List of trading signals
        """
        pass
    
    def update(self, data: pd.DataFrame) -> List[Signal]:
        """
        Update strategy state and generate new signals.
        
        This method combines initialization check and signal generation.
        It's the main method called during backtesting.
        
        Args:
            data: Current market data
            
        Returns:
            List of trading signals
        """
        if not self.is_initialized:
            if not self.initialize(data):
                self.logger.error("Strategy initialization failed")
                return []
            self.is_initialized = True
        
        return self.generate_signals(data)
    
    def execute_trade(self, signal: Signal, current_price: float, 
                     available_cash: float) -> Optional[Trade]:
        """
        Execute a trade based on a signal.
        
        Args:
            signal: Trading signal to execute
            current_price: Current market price
            available_cash: Available cash for trading
            
        Returns:
            Trade object if execution was successful, None otherwise
        """
        if signal.signal_type not in ['buy', 'sell']:
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, current_price, available_cash)
        if position_size <= 0:
            return None
        
        # Create trade
        trade = Trade(
            trade_id=f"{self.config.name}_{len(self.trades)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy_name=self.config.name,
            symbol=signal.symbol,
            entry_time=signal.timestamp,
            entry_price=current_price,
            quantity=position_size,
            side='long' if signal.signal_type == 'buy' else 'short',
            entry_value=position_size * current_price,
            entry_signal=signal
        )
        
        # Set stop loss and take profit
        if self.config.stop_loss_pct > 0:
            if signal.signal_type == 'buy':
                trade.stop_loss = current_price * (1 - self.config.stop_loss_pct)
            else:
                trade.stop_loss = current_price * (1 + self.config.stop_loss_pct)
        
        if self.config.take_profit_pct > 0:
            if signal.signal_type == 'buy':
                trade.take_profit = current_price * (1 + self.config.take_profit_pct)
            else:
                trade.take_profit = current_price * (1 - self.config.take_profit_pct)
        
        # Add to trades list
        self.trades.append(trade)
        self.current_position = trade
        
        self.logger.info(f"Executed {signal.signal_type} trade: {trade.quantity} {trade.symbol} at {current_price}")
        return trade
    
    def close_trade(self, trade: Trade, exit_price: float, 
                   exit_time: datetime, reason: str = "signal") -> bool:
        """
        Close an open trade.
        
        Args:
            trade: Trade to close
            exit_price: Exit price
            exit_time: Exit time
            reason: Reason for closing
            
        Returns:
            True if trade was closed successfully, False otherwise
        """
        if trade.status != "open":
            return False
        
        # Update trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_value = trade.quantity * exit_price
        trade.status = "closed"
        trade.close_reason = reason
        
        # Calculate PnL
        if trade.side == 'long':
            trade.pnl = trade.exit_value - trade.entry_value
        else:
            trade.pnl = trade.entry_value - trade.exit_value
        
        trade.pnl_pct = trade.pnl / trade.entry_value
        
        # Update current position
        if self.current_position and self.current_position.trade_id == trade.trade_id:
            self.current_position = None
        
        self.logger.info(f"Closed trade {trade.trade_id}: PnL = {trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        return True
    
    def _calculate_position_size(self, signal: Signal, current_price: float, 
                                available_cash: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            available_cash: Available cash for trading
            
        Returns:
            Position size in base currency
        """
        # Use risk manager if available
        if self.risk_manager:
            return self.risk_manager.calculate_position_size(
                signal, current_price, available_cash, self.config
            )
        
        # Default position sizing
        max_position_value = available_cash * self.config.max_position_size
        position_size = max_position_value / current_price
        
        return position_size
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics for the strategy.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.status == "closed"]
        open_trades = [t for t in self.trades if t.status == "open"]
        
        if not closed_trades:
            return {
                'total_trades': total_trades,
                'open_trades': len(open_trades),
                'closed_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        
        # PnL metrics
        pnls = [t.pnl for t in closed_trades if t.pnl is not None]
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls) if pnls else 0.0
        
        # Win rate
        winning_trades = [p for p in pnls if p > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
        
        # Profit factor
        total_profit = sum([p for p in pnls if p > 0])
        total_loss = abs(sum([p for p in pnls if p < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'open_trades': len(open_trades),
            'closed_trades': len(closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'largest_win': max(pnls) if pnls else 0.0,
            'largest_loss': min(pnls) if pnls else 0.0,
            'current_position': self.current_position is not None
        }
    
    def reset(self):
        """Reset strategy state for new backtest"""
        self.signals.clear()
        self.trades.clear()
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        self.current_position = None
        self.is_initialized = False
    
    def set_risk_manager(self, risk_manager: BaseRiskManager):
        """Set the risk manager for this strategy"""
        self.risk_manager = risk_manager
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return {
            'name': self.config.name,
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'max_position_size': self.config.max_position_size,
            'stop_loss_pct': self.config.stop_loss_pct,
            'take_profit_pct': self.config.take_profit_pct,
            **self.config.parameters
        }
    
    def update_parameters(self, new_params: Dict[str, Any]):
        """Update strategy parameters"""
        for key, value in new_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key in self.config.parameters:
                self.config.parameters[key] = value
        
        self.logger.info(f"Updated parameters: {new_params}")
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return f"{self.config.name}({self.config.symbol}, {self.config.timeframe})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(name='{self.config.name}', symbol='{self.config.symbol}')"
