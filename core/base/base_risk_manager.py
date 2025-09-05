"""
Abstract Base Risk Manager for Backtesting

This module provides the abstract base class for all risk managers.
All risk managers must implement the methods defined in this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .base_strategy import Signal, StrategyConfig


@dataclass
class RiskConfig:
    """Configuration for risk management"""
    
    # Portfolio risk limits
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_correlation: float = 0.7      # Max correlation between positions
    max_drawdown: float = 0.15        # 15% max drawdown limit
    max_leverage: float = 2.0         # Max leverage allowed
    
    # Position sizing
    position_sizing_method: str = "kelly"  # kelly, kelly_vol, equal_risk, fixed
    fixed_position_size: float = 0.1       # Fixed position size (10% of portfolio)
    
    # Stop loss and take profit
    default_stop_loss: float = 0.05        # 5% default stop loss
    default_take_profit: float = 0.10      # 10% default take profit
    trailing_stop: bool = True
    trailing_stop_distance: float = 0.02   # 2% trailing stop distance
    
    # Volatility-based adjustments
    volatility_lookback: int = 20          # Periods for volatility calculation
    volatility_adjustment: bool = True     # Adjust position size based on volatility
    min_volatility: float = 0.01          # Minimum volatility threshold
    max_volatility: float = 0.50          # Maximum volatility threshold
    
    # Risk metrics
    var_confidence: float = 0.95          # VaR confidence level
    cvar_confidence: float = 0.95         # CVaR confidence level
    stress_test_scenarios: int = 1000     # Number of stress test scenarios
    
    # Correlation management
    correlation_lookback: int = 60         # Periods for correlation calculation
    correlation_threshold: float = 0.7     # Correlation threshold for position limits
    
    # Dynamic adjustments
    enable_dynamic_adjustment: bool = True
    adjustment_frequency: str = "daily"    # daily, weekly, monthly
    performance_threshold: float = 0.02    # Performance threshold for adjustments


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Risk metrics
    var_95: float                          # Value at Risk at 95% confidence
    cvar_95: float                         # Conditional Value at Risk
    volatility: float                      # Position volatility
    beta: float                            # Beta relative to market
    
    # Risk limits
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Metadata
    entry_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioRisk:
    """Risk metrics for the entire portfolio"""
    
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    
    # Risk metrics
    portfolio_var: float                   # Portfolio Value at Risk
    portfolio_cvar: float                  # Portfolio Conditional Value at Risk
    portfolio_volatility: float            # Portfolio volatility
    max_drawdown: float                    # Maximum drawdown
    
    # Position metrics
    total_positions: int
    long_positions: int
    short_positions: int
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Risk limits
    current_risk: float                    # Current portfolio risk
    risk_limit: float                      # Risk limit
    risk_utilization: float                # Risk utilization percentage
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)


class BaseRiskManager(ABC):
    """
    Abstract base class for all risk managers.
    
    This class defines the interface that all risk managers must implement.
    It provides common functionality for risk assessment, position sizing,
    and portfolio risk management.
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize the base risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_history: List[PortfolioRisk] = []
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the risk manager"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_config(self):
        """Validate risk manager configuration"""
        if self.config.max_portfolio_risk <= 0 or self.config.max_portfolio_risk > 1:
            raise ValueError("Max portfolio risk must be between 0 and 1")
        
        if self.config.max_drawdown <= 0 or self.config.max_drawdown > 1:
            raise ValueError("Max drawdown must be between 0 and 1")
        
        if self.config.max_leverage <= 0:
            raise ValueError("Max leverage must be positive")
        
        if self.config.var_confidence <= 0 or self.config.var_confidence >= 1:
            raise ValueError("VaR confidence must be between 0 and 1")
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float, strategy_config: StrategyConfig) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            available_cash: Available cash for trading
            strategy_config: Strategy configuration
            
        Returns:
            Optimal position size in base currency
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, entry_price: float, signal: Signal, 
                           volatility: float) -> float:
        """
        Calculate optimal stop loss level.
        
        Args:
            entry_price: Entry price for the position
            signal: Trading signal
            volatility: Current market volatility
            
        Returns:
            Stop loss price level
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(self, entry_price: float, signal: Signal, 
                             volatility: float) -> float:
        """
        Calculate optimal take profit level.
        
        Args:
            entry_price: Entry price for the position
            signal: Trading signal
            volatility: Current market volatility
            
        Returns:
            Take profit price level
        """
        pass
    
    @abstractmethod
    def assess_position_risk(self, symbol: str, position_size: float, 
                            entry_price: float, current_price: float, 
                            market_data: pd.DataFrame) -> PositionRisk:
        """
        Assess risk for a single position.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            entry_price: Entry price
            current_price: Current market price
            market_data: Market data for risk calculation
            
        Returns:
            PositionRisk object with risk metrics
        """
        pass
    
    @abstractmethod
    def assess_portfolio_risk(self, positions: Dict[str, PositionRisk], 
                             market_data: Dict[str, pd.DataFrame]) -> PortfolioRisk:
        """
        Assess risk for the entire portfolio.
        
        Args:
            positions: Dictionary of current positions
            market_data: Market data for all symbols
            
        Returns:
            PortfolioRisk object with portfolio risk metrics
        """
        pass
    
    def add_position(self, symbol: str, position_size: float, entry_price: float, 
                    market_data: pd.DataFrame):
        """
        Add a new position to risk management.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            entry_price: Entry price
            market_data: Market data for risk calculation
        """
        position_risk = self.assess_position_risk(
            symbol, position_size, entry_price, entry_price, market_data
        )
        self.positions[symbol] = position_risk
        self.logger.info(f"Added position: {symbol}, size: {position_size}, price: {entry_price}")
    
    def update_position(self, symbol: str, current_price: float, 
                       market_data: pd.DataFrame):
        """
        Update an existing position with current market data.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            market_data: Current market data
        """
        if symbol not in self.positions:
            self.logger.warning(f"Position {symbol} not found for update")
            return
        
        position = self.positions[symbol]
        position.current_price = current_price
        position.unrealized_pnl = (current_price - position.entry_price) * position.position_size
        position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
        position.last_updated = datetime.now()
        
        # Update risk metrics
        updated_risk = self.assess_position_risk(
            symbol, position.position_size, position.entry_price, 
            current_price, market_data
        )
        
        # Update risk metrics
        position.var_95 = updated_risk.var_95
        position.cvar_95 = updated_risk.cvar_95
        position.volatility = updated_risk.volatility
        position.beta = updated_risk.beta
    
    def remove_position(self, symbol: str):
        """
        Remove a position from risk management.
        
        Args:
            symbol: Trading symbol to remove
        """
        if symbol in self.positions:
            del self.positions[symbol]
            self.logger.info(f"Removed position: {symbol}")
        else:
            self.logger.warning(f"Position {symbol} not found for removal")
    
    def check_risk_limits(self, new_position: PositionRisk) -> Tuple[bool, List[str]]:
        """
        Check if a new position would violate risk limits.
        
        Args:
            new_position: Proposed new position
            
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        violations = []
        
        # Check position size limits
        if new_position.position_size > self.config.fixed_position_size:
            violations.append(f"Position size {new_position.position_size:.2%} exceeds limit {self.config.fixed_position_size:.2%}")
        
        # Check correlation limits
        if len(self.positions) > 0:
            correlation_violations = self._check_correlation_limits(new_position)
            violations.extend(correlation_violations)
        
        # Check portfolio risk limits
        if self._calculate_portfolio_risk_with_new_position(new_position) > self.config.max_portfolio_risk:
            violations.append("New position would exceed portfolio risk limit")
        
        return len(violations) == 0, violations
    
    def _check_correlation_limits(self, new_position: PositionRisk) -> List[str]:
        """Check correlation limits for a new position"""
        violations = []
        
        # This is a simplified check - in practice, you'd need market data
        # to calculate actual correlations
        if len(self.positions) > 0:
            # Simulate correlation check
            if np.random.random() > 0.5:  # Placeholder
                violations.append("New position would exceed correlation limits")
        
        return violations
    
    def _calculate_portfolio_risk_with_new_position(self, new_position: PositionRisk) -> float:
        """Calculate portfolio risk including a new position"""
        # This is a simplified calculation
        # In practice, you'd use proper portfolio risk models
        
        current_risk = sum(pos.var_95 for pos in self.positions.values())
        new_risk = new_position.var_95
        
        return current_risk + new_risk
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get summary of current portfolio risk.
        
        Returns:
            Dictionary with portfolio risk summary
        """
        if not self.positions:
            return {
                'total_positions': 0,
                'total_value': 0.0,
                'total_pnl': 0.0,
                'portfolio_risk': 0.0
            }
        
        total_value = sum(pos.position_size * pos.current_price for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        portfolio_risk = sum(pos.var_95 for pos in self.positions.values())
        
        return {
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'portfolio_risk': portfolio_risk,
            'positions': {
                symbol: {
                    'size': pos.position_size,
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': pos.unrealized_pnl_pct,
                    'var_95': pos.var_95
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, 
                                    avg_loss: float, available_cash: float) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            available_cash: Available cash for trading
            
        Returns:
            Kelly position size
        """
        if avg_loss == 0:
            return 0.0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply Kelly fraction with risk limits
        kelly_fraction = max(0.0, min(kelly_fraction, self.config.max_portfolio_risk))
        
        return available_cash * kelly_fraction
    
    def calculate_volatility_adjusted_position_size(self, base_size: float, 
                                                  volatility: float) -> float:
        """
        Adjust position size based on volatility.
        
        Args:
            base_size: Base position size
            volatility: Current volatility
            
        Returns:
            Volatility-adjusted position size
        """
        if not self.config.volatility_adjustment:
            return base_size
        
        # Normalize volatility to reasonable range
        volatility = max(self.config.min_volatility, 
                        min(volatility, self.config.max_volatility))
        
        # Inverse relationship: higher volatility = smaller position
        adjustment_factor = self.config.min_volatility / volatility
        
        return base_size * adjustment_factor
    
    def stress_test_portfolio(self, market_data: Dict[str, pd.DataFrame], 
                             scenarios: int = None) -> Dict[str, Any]:
        """
        Perform stress testing on the portfolio.
        
        Args:
            market_data: Market data for all symbols
            scenarios: Number of stress test scenarios
            
        Returns:
            Dictionary with stress test results
        """
        if scenarios is None:
            scenarios = self.config.stress_test_scenarios
        
        # This is a simplified stress test
        # In practice, you'd implement proper Monte Carlo simulation
        
        results = {
            'scenarios': scenarios,
            'worst_case_loss': 0.0,
            'expected_shortfall': 0.0,
            'risk_metrics': {}
        }
        
        # Simulate stress test scenarios
        for i in range(scenarios):
            # Simulate market shock
            shock_factor = np.random.normal(0, 0.1)  # 10% volatility shock
            
            scenario_loss = 0.0
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    # Calculate loss under stress scenario
                    price_shock = position.current_price * (1 + shock_factor)
                    scenario_loss += (position.entry_price - price_shock) * position.position_size
            
            results['worst_case_loss'] = min(results['worst_case_loss'], scenario_loss)
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up risk manager resources")
        self.positions.clear()
        self.portfolio_history.clear()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
