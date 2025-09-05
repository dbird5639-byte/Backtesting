"""
Advanced Risk Management System
Based on methodologies from AI projects for comprehensive portfolio risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    CASH = "cash"

@dataclass
class Position:
    """Enhanced position data structure"""
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    last_updated: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    position_value: float = 0.0
    weight: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # Portfolio metrics
    total_value: float
    total_exposure: float
    leverage_ratio: float
    cash_ratio: float
    
    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Volatility metrics
    portfolio_volatility: float
    systematic_risk: float
    idiosyncratic_risk: float
    
    # Correlation metrics
    avg_correlation: float
    max_correlation: float
    diversification_ratio: float
    
    # Concentration metrics
    herfindahl_index: float
    max_position_weight: float
    effective_n: float  # Effective number of positions

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    timestamp: datetime
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    current_value: float
    threshold_value: float
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedRiskManager:
    """
    Advanced risk management system incorporating AI project methodologies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.risk_level = RiskLevel(self.config.get('risk_level', 'moderate'))
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.06)
        self.max_position_weight = self.config.get('max_position_weight', 0.1)
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.20)
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash = self.config.get('initial_cash', 100000.0)
        self.total_value = self.cash
        self.peak_value = self.cash
        
        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Risk limits based on risk level
        self.risk_limits = self._get_risk_limits()
        
        self.logger.info(f"Advanced Risk Manager initialized with {self.risk_level.value} risk level")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'risk_level': 'moderate',
            'max_portfolio_risk': 0.06,
            'max_position_weight': 0.1,
            'max_correlation': 0.7,
            'max_drawdown_limit': 0.20,
            'var_confidence': 0.95,
            'rebalance_frequency': 'daily',
            'correlation_window': 252,
            'volatility_window': 60,
            'initial_cash': 100000.0,
            'min_position_size': 1000.0,
            'max_leverage': 2.0,
            'liquidity_threshold': 0.8
        }
    
    def _get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits based on risk level"""
        limits = {
            'conservative': {
                'max_position_weight': 0.05,
                'max_portfolio_risk': 0.03,
                'max_drawdown_limit': 0.10,
                'max_leverage': 1.0,
                'var_threshold': 0.02
            },
            'moderate': {
                'max_position_weight': 0.10,
                'max_portfolio_risk': 0.06,
                'max_drawdown_limit': 0.20,
                'max_leverage': 2.0,
                'var_threshold': 0.04
            },
            'aggressive': {
                'max_position_weight': 0.15,
                'max_portfolio_risk': 0.10,
                'max_drawdown_limit': 0.30,
                'max_leverage': 3.0,
                'var_threshold': 0.06
            },
            'very_aggressive': {
                'max_position_weight': 0.20,
                'max_portfolio_risk': 0.15,
                'max_drawdown_limit': 0.40,
                'max_leverage': 5.0,
                'var_threshold': 0.08
            }
        }
        return limits.get(self.risk_level.value, limits['moderate'])
    
    def add_position(self, symbol: str, position_type: PositionType, quantity: float, 
                    price: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new position to the portfolio"""
        try:
            if symbol in self.positions:
                self.logger.warning(f"Position {symbol} already exists")
                return False
            
            # Calculate position value
            position_value = abs(quantity * price)
            
            # Check position size limits
            if position_value < self.config['min_position_size']:
                self.logger.warning(f"Position size too small: {position_value}")
                return False
            
            # Check if we have enough cash for long positions
            if position_type == PositionType.LONG and position_value > self.cash:
                self.logger.warning(f"Insufficient cash for position: {position_value} > {self.cash}")
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                position_type=position_type,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                entry_time=datetime.now(),
                last_updated=datetime.now(),
                position_value=position_value,
                weight=position_value / self.total_value,
                metadata=metadata or {}
            )
            
            self.positions[symbol] = position
            
            # Update cash
            if position_type == PositionType.LONG:
                self.cash -= position_value
            elif position_type == PositionType.SHORT:
                self.cash += position_value  # Short positions add cash
            
            # Update total value
            self._update_portfolio_value()
            
            self.logger.info(f"Added position: {symbol} {position_type.value} {quantity} @ {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float, 
                       volatility: Optional[float] = None, beta: Optional[float] = None):
        """Update position with current market data"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"Position {symbol} not found")
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            position.last_updated = datetime.now()
            
            # Update position value
            position.position_value = abs(position.quantity * current_price)
            
            # Calculate P&L
            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            elif position.position_type == PositionType.SHORT:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            # Update volatility and beta
            if volatility is not None:
                position.volatility = volatility
            if beta is not None:
                position.beta = beta
            
            # Update portfolio value
            self._update_portfolio_value()
            
            # Update position weight
            position.weight = position.position_value / self.total_value
            
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        # Calculate total position value
        total_position_value = sum(pos.position_value for pos in self.positions.values())
        
        # For short positions, we need to account for the liability
        short_liability = sum(
            pos.position_value for pos in self.positions.values() 
            if pos.position_type == PositionType.SHORT
        )
        
        # Total value = cash + long positions - short liability
        self.total_value = self.cash + total_position_value - short_liability
        
        # Update peak value
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
    
    def calculate_risk_metrics(self, returns_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Basic portfolio metrics
            total_exposure = sum(pos.position_value for pos in self.positions.values())
            leverage_ratio = total_exposure / self.total_value if self.total_value > 0 else 0
            cash_ratio = self.cash / self.total_value if self.total_value > 0 else 1
            
            # Calculate returns if not provided
            if returns_data is None:
                returns_data = self._calculate_portfolio_returns()
            
            # Risk metrics
            var_95, var_99 = self._calculate_var(returns_data, [0.95, 0.99])
            cvar_95, cvar_99 = self._calculate_cvar(returns_data, [0.95, 0.99])
            max_drawdown, current_drawdown = self._calculate_drawdowns()
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns_data)
            sortino_ratio = self._calculate_sortino_ratio(returns_data)
            calmar_ratio = self._calculate_calmar_ratio(returns_data)
            information_ratio = self._calculate_information_ratio(returns_data)
            
            # Volatility metrics
            portfolio_volatility = returns_data.std() * np.sqrt(252) if len(returns_data) > 0 else 0
            systematic_risk, idiosyncratic_risk = self._calculate_risk_decomposition()
            
            # Correlation metrics
            avg_correlation, max_correlation, diversification_ratio = self._calculate_correlation_metrics()
            
            # Concentration metrics
            herfindahl_index, max_position_weight, effective_n = self._calculate_concentration_metrics()
            
            return RiskMetrics(
                total_value=self.total_value,
                total_exposure=total_exposure,
                leverage_ratio=leverage_ratio,
                cash_ratio=cash_ratio,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                portfolio_volatility=portfolio_volatility,
                systematic_risk=systematic_risk,
                idiosyncratic_risk=idiosyncratic_risk,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                diversification_ratio=diversification_ratio,
                herfindahl_index=herfindahl_index,
                max_position_weight=max_position_weight,
                effective_n=effective_n
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return self._empty_risk_metrics()
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns from performance history"""
        if len(self.performance_history) < 2:
            return pd.Series(dtype=float)
        
        # Calculate daily returns from performance history
        values = [entry['total_value'] for entry in self.performance_history]
        returns = pd.Series(values).pct_change().dropna()
        return returns
    
    def _calculate_var(self, returns: pd.Series, confidence_levels: List[float]) -> Tuple[float, float]:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        var_values = []
        for conf in confidence_levels:
            var = np.percentile(returns, (1 - conf) * 100)
            var_values.append(var)
        
        return tuple(var_values)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_levels: List[float]) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        cvar_values = []
        for conf in confidence_levels:
            var = np.percentile(returns, (1 - conf) * 100)
            cvar = returns[returns <= var].mean()
            cvar_values.append(cvar)
        
        return tuple(cvar_values)
    
    def _calculate_drawdowns(self) -> Tuple[float, float]:
        """Calculate maximum and current drawdowns"""
        if len(self.performance_history) < 2:
            return 0.0, 0.0
        
        values = [entry['total_value'] for entry in self.performance_history]
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        
        max_drawdown = np.min(drawdowns)
        current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0.0
        
        return max_drawdown, current_drawdown
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        risk_free_rate = 0.02
        excess_returns = returns.mean() - risk_free_rate / 252
        downside_deviation = downside_returns.std()
        
        return excess_returns / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd, _ = self._calculate_drawdowns()
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information ratio (simplified)"""
        if len(returns) == 0:
            return 0.0
        
        # Assume benchmark return of 0 for simplicity
        benchmark_return = 0.0
        excess_returns = returns - benchmark_return
        
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def _calculate_risk_decomposition(self) -> Tuple[float, float]:
        """Calculate systematic and idiosyncratic risk"""
        # Simplified calculation - would need market data for proper decomposition
        portfolio_vol = self._calculate_portfolio_volatility()
        systematic_risk = portfolio_vol * 0.7  # Assume 70% systematic
        idiosyncratic_risk = portfolio_vol * 0.3  # Assume 30% idiosyncratic
        
        return systematic_risk, idiosyncratic_risk
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if not self.positions:
            return 0.0
        
        # Calculate weighted average volatility
        total_value = sum(pos.position_value for pos in self.positions.values())
        if total_value == 0:
            return 0.0
        
        weighted_vol = sum(
            pos.volatility * (pos.position_value / total_value) 
            for pos in self.positions.values()
        )
        
        return weighted_vol
    
    def _calculate_correlation_metrics(self) -> Tuple[float, float, float]:
        """Calculate correlation metrics"""
        if len(self.positions) < 2:
            return 0.0, 0.0, 1.0
        
        # Simplified correlation calculation
        # In practice, would use historical returns data
        symbols = list(self.positions.keys())
        n = len(symbols)
        
        # Assume average correlation of 0.3 for simplicity
        avg_correlation = 0.3
        max_correlation = 0.5
        
        # Diversification ratio = weighted average vol / portfolio vol
        individual_vols = [pos.volatility for pos in self.positions.values()]
        portfolio_vol = self._calculate_portfolio_volatility()
        
        if portfolio_vol > 0:
            diversification_ratio = np.mean(individual_vols) / portfolio_vol
        else:
            diversification_ratio = 1.0
        
        return avg_correlation, max_correlation, diversification_ratio
    
    def _calculate_concentration_metrics(self) -> Tuple[float, float, float]:
        """Calculate concentration metrics"""
        if not self.positions:
            return 0.0, 0.0, 0.0
        
        weights = [pos.weight for pos in self.positions.values()]
        
        # Herfindahl index
        herfindahl_index = sum(w**2 for w in weights)
        
        # Maximum position weight
        max_position_weight = max(weights) if weights else 0.0
        
        # Effective number of positions
        effective_n = 1 / herfindahl_index if herfindahl_index > 0 else 0.0
        
        return herfindahl_index, max_position_weight, effective_n
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            total_value=self.total_value,
            total_exposure=0.0,
            leverage_ratio=0.0,
            cash_ratio=1.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            portfolio_volatility=0.0,
            systematic_risk=0.0,
            idiosyncratic_risk=0.0,
            avg_correlation=0.0,
            max_correlation=0.0,
            diversification_ratio=1.0,
            herfindahl_index=0.0,
            max_position_weight=0.0,
            effective_n=0.0
        )
    
    def check_risk_limits(self) -> List[RiskAlert]:
        """Check portfolio against risk limits and generate alerts"""
        alerts = []
        metrics = self.calculate_risk_metrics()
        
        # Check position weight limits
        for symbol, position in self.positions.items():
            if position.weight > self.risk_limits['max_position_weight']:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="position_weight",
                    severity="high",
                    message=f"Position {symbol} exceeds weight limit",
                    current_value=position.weight,
                    threshold_value=self.risk_limits['max_position_weight'],
                    recommendation=f"Reduce position size for {symbol}"
                )
                alerts.append(alert)
        
        # Check portfolio risk limits
        if abs(metrics.var_95) > self.risk_limits['var_threshold']:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="portfolio_risk",
                severity="critical",
                message="Portfolio VaR exceeds limit",
                current_value=abs(metrics.var_95),
                threshold_value=self.risk_limits['var_threshold'],
                recommendation="Reduce portfolio risk exposure"
            )
            alerts.append(alert)
        
        # Check drawdown limits
        if abs(metrics.current_drawdown) > self.risk_limits['max_drawdown_limit']:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="drawdown",
                severity="critical",
                message="Portfolio drawdown exceeds limit",
                current_value=abs(metrics.current_drawdown),
                threshold_value=self.risk_limits['max_drawdown_limit'],
                recommendation="Consider reducing positions or stopping trading"
            )
            alerts.append(alert)
        
        # Check leverage limits
        if metrics.leverage_ratio > self.risk_limits['max_leverage']:
            alert = RiskAlert(
                timestamp=datetime.now(),
                alert_type="leverage",
                severity="high",
                message="Portfolio leverage exceeds limit",
                current_value=metrics.leverage_ratio,
                threshold_value=self.risk_limits['max_leverage'],
                recommendation="Reduce leverage"
            )
            alerts.append(alert)
        
        # Store alerts
        self.risk_alerts.extend(alerts)
        
        return alerts
    
    def optimize_portfolio(self, target_return: Optional[float] = None) -> Dict[str, float]:
        """Optimize portfolio using modern portfolio theory"""
        try:
            if len(self.positions) < 2:
                return {}
            
            # Get position data
            symbols = list(self.positions.keys())
            weights = np.array([pos.weight for pos in self.positions.values()])
            volatilities = np.array([pos.volatility for pos in self.positions.values()])
            
            # Create correlation matrix (simplified)
            n = len(symbols)
            correlation_matrix = np.eye(n) * 0.3 + np.ones((n, n)) * 0.1
            
            # Calculate covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            # Objective function: minimize portfolio variance
            def objective(w):
                return np.dot(w, np.dot(cov_matrix, w))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds: each weight between 0 and max_position_weight
            bounds = [(0, self.risk_limits['max_position_weight']) for _ in range(n)]
            
            # Initial guess
            x0 = np.ones(n) / n
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                return dict(zip(symbols, optimal_weights))
            else:
                self.logger.warning("Portfolio optimization failed")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return {}
    
    def rebalance_portfolio(self, target_weights: Dict[str, float]) -> bool:
        """Rebalance portfolio to target weights"""
        try:
            if not target_weights:
                return False
            
            # Calculate required changes
            rebalance_orders = []
            
            for symbol, target_weight in target_weights.items():
                if symbol in self.positions:
                    current_weight = self.positions[symbol].weight
                    weight_change = target_weight - current_weight
                    
                    if abs(weight_change) > 0.01:  # Only rebalance if change > 1%
                        target_value = target_weight * self.total_value
                        current_value = self.positions[symbol].position_value
                        value_change = target_value - current_value
                        
                        rebalance_orders.append({
                            'symbol': symbol,
                            'current_weight': current_weight,
                            'target_weight': target_weight,
                            'value_change': value_change
                        })
            
            # Execute rebalancing (simplified - would need actual order execution)
            for order in rebalance_orders:
                self.logger.info(f"Rebalancing {order['symbol']}: {order['current_weight']:.3f} -> {order['target_weight']:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return False
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        metrics = self.calculate_risk_metrics()
        alerts = self.check_risk_limits()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_level': self.risk_level.value,
            'portfolio_metrics': {
                'total_value': metrics.total_value,
                'total_exposure': metrics.total_exposure,
                'leverage_ratio': metrics.leverage_ratio,
                'cash_ratio': metrics.cash_ratio
            },
            'risk_metrics': {
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'cvar_95': metrics.cvar_95,
                'cvar_99': metrics.cvar_99,
                'max_drawdown': metrics.max_drawdown,
                'current_drawdown': metrics.current_drawdown
            },
            'performance_metrics': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'information_ratio': metrics.information_ratio
            },
            'concentration_metrics': {
                'herfindahl_index': metrics.herfindahl_index,
                'max_position_weight': metrics.max_position_weight,
                'effective_n': metrics.effective_n
            },
            'active_alerts': len([a for a in alerts if a.severity in ['high', 'critical']]),
            'total_alerts': len(alerts),
            'positions': len(self.positions)
        }
    
    def update_performance_history(self):
        """Update performance history with current metrics"""
        entry = {
            'timestamp': datetime.now(),
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': sum(pos.position_value for pos in self.positions.values()),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
        }
        self.performance_history.append(entry)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = AdvancedRiskManager({
        'risk_level': 'moderate',
        'initial_cash': 100000.0
    })
    
    # Add some positions
    risk_manager.add_position('AAPL', PositionType.LONG, 100, 150.0)
    risk_manager.add_position('GOOGL', PositionType.LONG, 50, 2800.0)
    risk_manager.add_position('TSLA', PositionType.SHORT, 20, 200.0)
    
    # Update positions with current prices
    risk_manager.update_position('AAPL', 155.0, volatility=0.25, beta=1.2)
    risk_manager.update_position('GOOGL', 2850.0, volatility=0.30, beta=1.1)
    risk_manager.update_position('TSLA', 190.0, volatility=0.40, beta=1.5)
    
    # Update performance history
    risk_manager.update_performance_history()
    
    # Get risk report
    report = risk_manager.get_risk_report()
    print("Risk Report:")
    print(f"Total Value: ${report['portfolio_metrics']['total_value']:,.2f}")
    print(f"Leverage Ratio: {report['portfolio_metrics']['leverage_ratio']:.2f}")
    print(f"VaR 95%: {report['risk_metrics']['var_95']:.3f}")
    print(f"Sharpe Ratio: {report['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"Active Alerts: {report['active_alerts']}")
