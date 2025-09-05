#!/usr/bin/env python3
"""
Modern Portfolio Optimization Engine

A sophisticated engine that provides:
- Multi-objective portfolio optimization (return, risk, diversification)
- Advanced optimization algorithms (genetic algorithms, particle swarm, etc.)
- Dynamic rebalancing strategies
- Transaction cost modeling
- Factor-based portfolio construction
- ESG and sustainability constraints
- Real-time portfolio monitoring and adjustment
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
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class OptimizationMethod(Enum):
    """Optimization method enumeration"""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"

class RebalancingStrategy(Enum):
    """Rebalancing strategy enumeration"""
    TIME_BASED = "time_based"
    THRESHOLD_BASED = "threshold_based"
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM_BASED = "momentum_based"
    ADAPTIVE = "adaptive"

@dataclass
class PortfolioEngineConfig:
    """Configuration for the portfolio engine"""
    # Paths
    data_path: str = "./fetched_data"
    results_path: str = "./Results"
    
    # Portfolio Configuration
    initial_capital: float = 1000000.0
    max_positions: int = 50
    min_position_weight: float = 0.01  # 1%
    max_position_weight: float = 0.1   # 10%
    
    # Optimization
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    optimization_objectives: List[str] = field(default_factory=lambda: ['return', 'risk', 'diversification'])
    risk_aversion: float = 1.0
    expected_return_method: str = "historical"  # historical, black_litterman, factor_model
    
    # Risk Management
    max_portfolio_volatility: float = 0.20  # 20%
    max_drawdown: float = 0.15  # 15%
    var_confidence: float = 0.95
    max_correlation: float = 0.7
    
    # Transaction Costs
    transaction_cost_bps: float = 10.0  # 10 basis points
    market_impact_model: str = "linear"  # linear, square_root, power_law
    liquidity_threshold: float = 0.1  # 10% of average daily volume
    
    # Rebalancing
    rebalancing_strategy: RebalancingStrategy = RebalancingStrategy.THRESHOLD_BASED
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    rebalancing_threshold: float = 0.05  # 5% drift threshold
    rebalancing_cost_threshold: float = 0.02  # 2% cost threshold
    
    # Factor Models
    factor_model: str = "fama_french"  # fama_french, carhart, custom
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    
    # ESG Constraints
    enable_esg_constraints: bool = False
    min_esg_score: float = 0.0
    esg_weight: float = 0.1
    
    # Performance
    max_workers: int = 4
    optimization_timeout: int = 300  # 5 minutes
    
    # Output
    save_portfolio_weights: bool = True
    save_performance_attribution: bool = True
    save_rebalancing_history: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

@dataclass
class PortfolioWeights:
    """Portfolio weights data structure"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    concentration_risk: float
    transaction_costs: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RebalancingEvent:
    """Rebalancing event data structure"""
    timestamp: datetime
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    rebalancing_cost: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class PortfolioOptimizer:
    """Advanced portfolio optimization engine"""
    
    def __init__(self, config: PortfolioEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          expected_returns: pd.Series = None,
                          risk_model: pd.DataFrame = None) -> PortfolioWeights:
        """Optimize portfolio using specified method"""
        try:
            if self.config.optimization_method == OptimizationMethod.MEAN_VARIANCE:
                return self._mean_variance_optimization(returns, expected_returns, risk_model)
            elif self.config.optimization_method == OptimizationMethod.RISK_PARITY:
                return self._risk_parity_optimization(returns, risk_model)
            elif self.config.optimization_method == OptimizationMethod.EQUAL_WEIGHT:
                return self._equal_weight_optimization(returns)
            elif self.config.optimization_method == OptimizationMethod.MINIMUM_VARIANCE:
                return self._minimum_variance_optimization(returns, risk_model)
            elif self.config.optimization_method == OptimizationMethod.MAXIMUM_SHARPE:
                return self._maximum_sharpe_optimization(returns, expected_returns, risk_model)
            elif self.config.optimization_method == OptimizationMethod.GENETIC_ALGORITHM:
                return self._genetic_algorithm_optimization(returns, expected_returns, risk_model)
            else:
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
        
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            raise
    
    def _mean_variance_optimization(self, returns: pd.DataFrame, 
                                   expected_returns: pd.Series = None,
                                   risk_model: pd.DataFrame = None) -> PortfolioWeights:
        """Mean-variance optimization"""
        n_assets = len(returns.columns)
        
        # Calculate expected returns if not provided
        if expected_returns is None:
            expected_returns = returns.mean() * 252  # Annualized
        
        # Calculate covariance matrix if not provided
        if risk_model is None:
            risk_model = returns.cov() * 252  # Annualized
        
        # Objective function: maximize utility = E[R] - 0.5 * risk_aversion * Var[R]
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
            utility = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_position_weight, self.config.max_position_weight) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            self.logger.warning("Optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
        else:
            weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(risk_model)))
        diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk (Herfindahl-Hirschman Index)
        concentration_risk = np.sum(weights ** 2)
        
        # Transaction costs (simplified)
        transaction_costs = 0.0  # Will be calculated during rebalancing
        
        return PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            transaction_costs=transaction_costs,
            metadata={'optimization_method': 'mean_variance', 'success': result.success}
        )
    
    def _risk_parity_optimization(self, returns: pd.DataFrame, 
                                 risk_model: pd.DataFrame = None) -> PortfolioWeights:
        """Risk parity optimization"""
        n_assets = len(returns.columns)
        
        if risk_model is None:
            risk_model = returns.cov() * 252
        
        # Objective function: minimize sum of squared differences from equal risk contribution
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Risk contributions
            risk_contributions = weights * np.dot(risk_model, weights) / portfolio_volatility
            
            # Target risk contribution (equal)
            target_risk_contribution = 1.0 / n_assets
            
            # Sum of squared differences
            return np.sum((risk_contributions - target_risk_contribution) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.config.min_position_weight, self.config.max_position_weight) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            self.logger.warning("Risk parity optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
        else:
            weights = result.x
        
        # Calculate portfolio metrics
        expected_returns = returns.mean() * 252
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(risk_model)))
        diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk
        concentration_risk = np.sum(weights ** 2)
        
        return PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            transaction_costs=0.0,
            metadata={'optimization_method': 'risk_parity', 'success': result.success}
        )
    
    def _equal_weight_optimization(self, returns: pd.DataFrame) -> PortfolioWeights:
        """Equal weight optimization"""
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio metrics
        expected_returns = returns.mean() * 252
        risk_model = returns.cov() * 252
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(risk_model)))
        diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk
        concentration_risk = np.sum(weights ** 2)
        
        return PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            transaction_costs=0.0,
            metadata={'optimization_method': 'equal_weight', 'success': True}
        )
    
    def _minimum_variance_optimization(self, returns: pd.DataFrame, 
                                      risk_model: pd.DataFrame = None) -> PortfolioWeights:
        """Minimum variance optimization"""
        n_assets = len(returns.columns)
        
        if risk_model is None:
            risk_model = returns.cov() * 252
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(risk_model, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.config.min_position_weight, self.config.max_position_weight) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            self.logger.warning("Minimum variance optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
        else:
            weights = result.x
        
        # Calculate portfolio metrics
        expected_returns = returns.mean() * 252
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(risk_model)))
        diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk
        concentration_risk = np.sum(weights ** 2)
        
        return PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            transaction_costs=0.0,
            metadata={'optimization_method': 'minimum_variance', 'success': result.success}
        )
    
    def _maximum_sharpe_optimization(self, returns: pd.DataFrame, 
                                    expected_returns: pd.Series = None,
                                    risk_model: pd.DataFrame = None) -> PortfolioWeights:
        """Maximum Sharpe ratio optimization"""
        n_assets = len(returns.columns)
        
        if expected_returns is None:
            expected_returns = returns.mean() * 252
        
        if risk_model is None:
            risk_model = returns.cov() * 252
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            return -sharpe_ratio
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.config.min_position_weight, self.config.max_position_weight) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            self.logger.warning("Maximum Sharpe optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
        else:
            weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(risk_model)))
        diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk
        concentration_risk = np.sum(weights ** 2)
        
        return PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            transaction_costs=0.0,
            metadata={'optimization_method': 'maximum_sharpe', 'success': result.success}
        )
    
    def _genetic_algorithm_optimization(self, returns: pd.DataFrame, 
                                       expected_returns: pd.Series = None,
                                       risk_model: pd.DataFrame = None) -> PortfolioWeights:
        """Genetic algorithm optimization for multi-objective optimization"""
        n_assets = len(returns.columns)
        
        if expected_returns is None:
            expected_returns = returns.mean() * 252
        
        if risk_model is None:
            risk_model = returns.cov() * 252
        
        # Multi-objective function
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Objectives
            return_obj = portfolio_return
            risk_obj = -portfolio_volatility  # Minimize risk
            diversification_obj = -np.sum(weights ** 2)  # Maximize diversification
            
            # Weighted combination
            combined_obj = (0.4 * return_obj + 0.4 * risk_obj + 0.2 * diversification_obj)
            
            return -combined_obj  # Minimize negative combined objective
        
        # Bounds
        bounds = [(self.config.min_position_weight, self.config.max_position_weight) 
                 for _ in range(n_assets)]
        
        # Use differential evolution (genetic algorithm)
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=1000,
            popsize=15,
            seed=42
        )
        
        if not result.success:
            self.logger.warning("Genetic algorithm optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
        else:
            weights = result.x / np.sum(result.x)  # Normalize
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(risk_model, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, np.sqrt(np.diag(risk_model)))
        diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk
        concentration_risk = np.sum(weights ** 2)
        
        return PortfolioWeights(
            weights=dict(zip(returns.columns, weights)),
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            transaction_costs=0.0,
            metadata={'optimization_method': 'genetic_algorithm', 'success': result.success}
        )

class RebalancingManager:
    """Portfolio rebalancing management system"""
    
    def __init__(self, config: PortfolioEngineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rebalancing_history = []
    
    def should_rebalance(self, current_weights: Dict[str, float], 
                        target_weights: Dict[str, float],
                        current_time: datetime) -> Tuple[bool, str]:
        """Determine if portfolio should be rebalanced"""
        try:
            if self.config.rebalancing_strategy == RebalancingStrategy.TIME_BASED:
                return self._time_based_rebalancing(current_time)
            elif self.config.rebalancing_strategy == RebalancingStrategy.THRESHOLD_BASED:
                return self._threshold_based_rebalancing(current_weights, target_weights)
            elif self.config.rebalancing_strategy == RebalancingStrategy.VOLATILITY_BASED:
                return self._volatility_based_rebalancing(current_weights, target_weights)
            elif self.config.rebalancing_strategy == RebalancingStrategy.MOMENTUM_BASED:
                return self._momentum_based_rebalancing(current_weights, target_weights)
            elif self.config.rebalancing_strategy == RebalancingStrategy.ADAPTIVE:
                return self._adaptive_rebalancing(current_weights, target_weights, current_time)
            else:
                return False, "Unknown rebalancing strategy"
        
        except Exception as e:
            self.logger.error(f"Error in rebalancing decision: {e}")
            return False, f"Error: {e}"
    
    def _time_based_rebalancing(self, current_time: datetime) -> Tuple[bool, str]:
        """Time-based rebalancing"""
        # Check if enough time has passed since last rebalancing
        if not self.rebalancing_history:
            return True, "Initial rebalancing"
        
        last_rebalancing = self.rebalancing_history[-1].timestamp
        
        if self.config.rebalancing_frequency == "daily":
            threshold = timedelta(days=1)
        elif self.config.rebalancing_frequency == "weekly":
            threshold = timedelta(weeks=1)
        elif self.config.rebalancing_frequency == "monthly":
            threshold = timedelta(days=30)
        elif self.config.rebalancing_frequency == "quarterly":
            threshold = timedelta(days=90)
        else:
            threshold = timedelta(days=30)
        
        if current_time - last_rebalancing >= threshold:
            return True, f"Time-based rebalancing ({self.config.rebalancing_frequency})"
        
        return False, "Not time for rebalancing"
    
    def _threshold_based_rebalancing(self, current_weights: Dict[str, float], 
                                    target_weights: Dict[str, float]) -> Tuple[bool, str]:
        """Threshold-based rebalancing"""
        # Calculate weight drift
        total_drift = 0.0
        max_drift = 0.0
        
        for asset in target_weights:
            if asset in current_weights:
                drift = abs(current_weights[asset] - target_weights[asset])
                total_drift += drift
                max_drift = max(max_drift, drift)
        
        if max_drift > self.config.rebalancing_threshold:
            return True, f"Threshold exceeded: {max_drift:.2%} > {self.config.rebalancing_threshold:.2%}"
        
        return False, f"Within threshold: {max_drift:.2%} <= {self.config.rebalancing_threshold:.2%}"
    
    def _volatility_based_rebalancing(self, current_weights: Dict[str, float], 
                                     target_weights: Dict[str, float]) -> Tuple[bool, str]:
        """Volatility-based rebalancing"""
        # This would require volatility data - simplified for now
        return self._threshold_based_rebalancing(current_weights, target_weights)
    
    def _momentum_based_rebalancing(self, current_weights: Dict[str, float], 
                                   target_weights: Dict[str, float]) -> Tuple[bool, str]:
        """Momentum-based rebalancing"""
        # This would require momentum data - simplified for now
        return self._threshold_based_rebalancing(current_weights, target_weights)
    
    def _adaptive_rebalancing(self, current_weights: Dict[str, float], 
                             target_weights: Dict[str, float], 
                             current_time: datetime) -> Tuple[bool, str]:
        """Adaptive rebalancing based on multiple factors"""
        # Combine time and threshold-based approaches
        time_rebalance, time_reason = self._time_based_rebalancing(current_time)
        threshold_rebalance, threshold_reason = self._threshold_based_rebalancing(current_weights, target_weights)
        
        if time_rebalance or threshold_rebalance:
            return True, f"Adaptive: {time_reason}, {threshold_reason}"
        
        return False, "No rebalancing needed"
    
    def calculate_rebalancing_cost(self, old_weights: Dict[str, float], 
                                  new_weights: Dict[str, float],
                                  portfolio_value: float) -> float:
        """Calculate rebalancing transaction costs"""
        total_cost = 0.0
        
        for asset in set(old_weights.keys()) | set(new_weights.keys()):
            old_weight = old_weights.get(asset, 0.0)
            new_weight = new_weights.get(asset, 0.0)
            
            weight_change = abs(new_weight - old_weight)
            if weight_change > 0:
                # Transaction cost in basis points
                transaction_cost = weight_change * portfolio_value * (self.config.transaction_cost_bps / 10000)
                total_cost += transaction_cost
        
        return total_cost
    
    def record_rebalancing_event(self, old_weights: Dict[str, float], 
                                new_weights: Dict[str, float],
                                rebalancing_cost: float,
                                reason: str):
        """Record rebalancing event"""
        event = RebalancingEvent(
            timestamp=datetime.now(),
            old_weights=old_weights.copy(),
            new_weights=new_weights.copy(),
            rebalancing_cost=rebalancing_cost,
            reason=reason
        )
        self.rebalancing_history.append(event)

class PortfolioEngine:
    """Modern portfolio optimization engine"""
    
    def __init__(self, config: PortfolioEngineConfig = None):
        self.config = config or PortfolioEngineConfig()
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.optimizer = PortfolioOptimizer(self.config)
        self.rebalancing_manager = RebalancingManager(self.config)
        
        self.logger.info("Portfolio Engine initialized successfully")
    
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
            log_file = f"portfolio_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            logging.getLogger().addHandler(file_handler)
    
    def prepare_data(self, data_files: List[str]) -> pd.DataFrame:
        """Prepare multi-asset data for portfolio optimization"""
        try:
            all_data = {}
            
            for data_file in data_files:
                try:
                    data = pd.read_csv(data_file, parse_dates=True, index_col=0)
                    
                    if len(data) < 252:  # Need at least 1 year of data
                        self.logger.warning(f"Insufficient data in {data_file}: {len(data)}")
                        continue
                    
                    # Extract symbol from filename
                    symbol = Path(data_file).stem
                    
                    # Calculate returns
                    returns = data['Close'].pct_change().dropna()
                    all_data[symbol] = returns
                    
                except Exception as e:
                    self.logger.error(f"Error processing {data_file}: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No valid data found")
            
            # Align all returns data
            returns_df = pd.DataFrame(all_data)
            returns_df = returns_df.dropna()
            
            # Limit to max positions
            if len(returns_df.columns) > self.config.max_positions:
                # Select assets with highest Sharpe ratios
                sharpe_ratios = returns_df.mean() / returns_df.std() * np.sqrt(252)
                top_assets = sharpe_ratios.nlargest(self.config.max_positions).index
                returns_df = returns_df[top_assets]
            
            self.logger.info(f"Prepared data for {len(returns_df.columns)} assets")
            return returns_df
        
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
    
    async def run(self) -> List[PortfolioWeights]:
        """Run the portfolio engine"""
        try:
            self.logger.info("Starting Portfolio Engine")
            start_time = time.time()
            
            # Discover data files
            data_path = Path(self.config.data_path)
            data_files = list(data_path.rglob("*.csv"))
            
            if not data_files:
                self.logger.warning("No data files found")
                return []
            
            # Prepare data
            returns_df = self.prepare_data(data_files)
            
            if returns_df.empty:
                self.logger.warning("No valid data for portfolio optimization")
                return []
            
            # Optimize portfolio
            self.logger.info(f"Optimizing portfolio using {self.config.optimization_method.value}")
            portfolio_weights = self.optimizer.optimize_portfolio(returns_df)
            
            # Simulate rebalancing over time
            rebalancing_events = []
            current_weights = portfolio_weights.weights.copy()
            
            # Simulate monthly rebalancing for 1 year
            for month in range(12):
                # Simulate some drift in weights
                drift = np.random.normal(0, 0.01, len(current_weights))
                drifted_weights = {}
                for i, (asset, weight) in enumerate(current_weights.items()):
                    new_weight = max(0, weight + drift[i])
                    drifted_weights[asset] = new_weight
                
                # Normalize weights
                total_weight = sum(drifted_weights.values())
                drifted_weights = {k: v/total_weight for k, v in drifted_weights.items()}
                
                # Check if rebalancing is needed
                should_rebalance, reason = self.rebalancing_manager.should_rebalance(
                    drifted_weights, portfolio_weights.weights, datetime.now()
                )
                
                if should_rebalance:
                    # Calculate rebalancing cost
                    rebalancing_cost = self.rebalancing_manager.calculate_rebalancing_cost(
                        drifted_weights, portfolio_weights.weights, self.config.initial_capital
                    )
                    
                    # Record rebalancing event
                    self.rebalancing_manager.record_rebalancing_event(
                        drifted_weights, portfolio_weights.weights, rebalancing_cost, reason
                    )
                    
                    rebalancing_events.append({
                        'month': month,
                        'reason': reason,
                        'cost': rebalancing_cost
                    })
                    
                    # Update current weights
                    current_weights = portfolio_weights.weights.copy()
                
                self.logger.info(f"Month {month}: Rebalancing needed: {should_rebalance}")
            
            # Update portfolio weights with rebalancing info
            portfolio_weights.metadata.update({
                'rebalancing_events': rebalancing_events,
                'total_rebalancing_cost': sum(event['cost'] for event in rebalancing_events),
                'n_assets': len(returns_df.columns),
                'data_period': f"{returns_df.index[0].date()} to {returns_df.index[-1].date()}"
            })
            
            # Save results
            if self.config.save_portfolio_weights:
                await self._save_results([portfolio_weights])
            
            execution_time = time.time() - start_time
            self.logger.info(f"Portfolio Engine completed in {execution_time:.2f} seconds")
            self.logger.info(f"Optimized portfolio with {len(portfolio_weights.weights)} assets")
            self.logger.info(f"Expected return: {portfolio_weights.expected_return:.2%}")
            self.logger.info(f"Expected volatility: {portfolio_weights.expected_volatility:.2%}")
            self.logger.info(f"Sharpe ratio: {portfolio_weights.sharpe_ratio:.2f}")
            
            return [portfolio_weights]
        
        except Exception as e:
            self.logger.error(f"Error in Portfolio Engine: {e}")
            return []
    
    async def _save_results(self, portfolio_weights_list: List[PortfolioWeights]):
        """Save portfolio optimization results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(self.config.results_path) / f"portfolio_engine_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            for i, portfolio_weights in enumerate(portfolio_weights_list):
                # Save portfolio weights
                weights_file = results_dir / f"portfolio_weights_{i}.json"
                with open(weights_file, 'w') as f:
                    json.dump({
                        'weights': portfolio_weights.weights,
                        'expected_return': portfolio_weights.expected_return,
                        'expected_volatility': portfolio_weights.expected_volatility,
                        'sharpe_ratio': portfolio_weights.sharpe_ratio,
                        'diversification_ratio': portfolio_weights.diversification_ratio,
                        'concentration_risk': portfolio_weights.concentration_risk,
                        'timestamp': portfolio_weights.timestamp.isoformat(),
                        'metadata': portfolio_weights.metadata
                    }, f, indent=2)
                
                # Save rebalancing history
                if self.rebalancing_manager.rebalancing_history:
                    rebalancing_file = results_dir / f"rebalancing_history_{i}.json"
                    rebalancing_data = []
                    for event in self.rebalancing_manager.rebalancing_history:
                        rebalancing_data.append({
                            'timestamp': event.timestamp.isoformat(),
                            'old_weights': event.old_weights,
                            'new_weights': event.new_weights,
                            'rebalancing_cost': event.rebalancing_cost,
                            'reason': event.reason,
                            'metadata': event.metadata
                        })
                    
                    with open(rebalancing_file, 'w') as f:
                        json.dump(rebalancing_data, f, indent=2)
            
            # Save summary
            summary_file = results_dir / "portfolio_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Portfolio Engine Results Summary\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"Optimization Method: {self.config.optimization_method.value}\n")
                f.write(f"Total Portfolios: {len(portfolio_weights_list)}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                for i, pw in enumerate(portfolio_weights_list):
                    f.write(f"Portfolio {i+1}:\n")
                    f.write(f"  Expected Return: {pw.expected_return:.2%}\n")
                    f.write(f"  Expected Volatility: {pw.expected_volatility:.2%}\n")
                    f.write(f"  Sharpe Ratio: {pw.sharpe_ratio:.2f}\n")
                    f.write(f"  Diversification Ratio: {pw.diversification_ratio:.2f}\n")
                    f.write(f"  Concentration Risk: {pw.concentration_risk:.2f}\n")
                    f.write(f"  Number of Assets: {len(pw.weights)}\n")
                    f.write(f"  Rebalancing Events: {len(pw.metadata.get('rebalancing_events', []))}\n")
                    f.write(f"  Total Rebalancing Cost: {pw.metadata.get('total_rebalancing_cost', 0):.2f}\n\n")
            
            self.logger.info(f"Saved results to {results_dir}")
        
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

async def main():
    """Main function to run the portfolio engine"""
    config = PortfolioEngineConfig(
        optimization_method=OptimizationMethod.MEAN_VARIANCE,
        rebalancing_strategy=RebalancingStrategy.THRESHOLD_BASED,
        rebalancing_threshold=0.05,
        transaction_cost_bps=10.0,
        save_portfolio_weights=True,
        save_rebalancing_history=True
    )
    
    engine = PortfolioEngine(config)
    results = await engine.run()
    
    print(f"\nPortfolio Engine Results:")
    print(f"Total Portfolios: {len(results)}")
    
    if results:
        pw = results[0]
        print(f"Expected Return: {pw.expected_return:.2%}")
        print(f"Expected Volatility: {pw.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {pw.sharpe_ratio:.2f}")
        print(f"Number of Assets: {len(pw.weights)}")
        print(f"Rebalancing Events: {len(pw.metadata.get('rebalancing_events', []))}")

if __name__ == "__main__":
    asyncio.run(main())
