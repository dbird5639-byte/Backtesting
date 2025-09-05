#!/usr/bin/env python3
"""
Optimized Portfolio Engine with Integrated Visualization
"""

import asyncio
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
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

from .base_optimized_engine import BaseOptimizedEngine, BaseEngineConfig

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
class PortfolioEngineConfig(BaseEngineConfig):
    """Configuration for Portfolio Engine"""
    # Optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.MAXIMUM_SHARPE
    rebalancing_strategy: RebalancingStrategy = RebalancingStrategy.THRESHOLD_BASED
    rebalancing_frequency: int = 30  # days
    rebalancing_threshold: float = 0.05  # 5% deviation threshold
    
    # Risk constraints
    max_weight_per_asset: float = 0.4
    min_weight_per_asset: float = 0.0
    max_sector_weight: float = 0.6
    max_turnover: float = 0.5
    
    # Transaction costs
    transaction_cost_rate: float = 0.001  # 0.1%
    min_trade_size: float = 0.01  # 1% of portfolio
    
    # Optimization parameters
    max_iterations: int = 1000
    population_size: int = 50
    convergence_threshold: float = 1e-6
    
    # Visualization settings
    generate_plots: bool = True
    plot_style: str = "seaborn-v0_8"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300

class OptimizedPortfolioEngine(BaseOptimizedEngine):
    """Optimized Portfolio Engine with integrated visualization"""
    
    def __init__(self, config: PortfolioEngineConfig = None):
        super().__init__(config or PortfolioEngineConfig())
        self.config: PortfolioEngineConfig = self.config
        self.portfolio_history = []
        self.optimization_results = {}
        
        # Setup visualization
        if self.config.generate_plots:
            plt.style.use(self.config.plot_style)
            sns.set_palette("husl")
    
    def optimize_portfolio(self, returns: pd.DataFrame, method: OptimizationMethod = None) -> Dict[str, Any]:
        """Optimize portfolio using specified method"""
        method = method or self.config.optimization_method
        
        try:
            if method == OptimizationMethod.MAXIMUM_SHARPE:
                return self._optimize_maximum_sharpe(returns)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                return self._optimize_minimum_variance(returns)
            elif method == OptimizationMethod.RISK_PARITY:
                return self._optimize_risk_parity(returns)
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                return self._optimize_equal_weight(returns)
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                return self._optimize_genetic_algorithm(returns)
            else:
                return self._optimize_mean_variance(returns)
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {}
    
    def _optimize_maximum_sharpe(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize for maximum Sharpe ratio"""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Objective function (negative Sharpe ratio)
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_std == 0:
                    return -np.inf
                return -(portfolio_return / portfolio_std)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                          for _ in range(len(expected_returns)))
            
            # Initial guess
            n_assets = len(expected_returns)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(negative_sharpe, x0, method='SLSQP', 
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': self.config.max_iterations})
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_std,
                    'sharpe_ratio': sharpe_ratio,
                    'method': 'maximum_sharpe',
                    'success': True
                }
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _optimize_minimum_variance(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize for minimum variance"""
        try:
            cov_matrix = returns.cov() * 252
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                          for _ in range(len(returns.columns)))
            
            n_assets = len(returns.columns)
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(portfolio_variance, x0, method='SLSQP',
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                portfolio_std = np.sqrt(portfolio_variance(weights))
                expected_returns = returns.mean() * 252
                portfolio_return = np.dot(weights, expected_returns)
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_std,
                    'sharpe_ratio': portfolio_return / portfolio_std if portfolio_std > 0 else 0,
                    'method': 'minimum_variance',
                    'success': True
                }
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize for risk parity"""
        try:
            cov_matrix = returns.cov() * 252
            
            def risk_parity_objective(weights):
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_std == 0:
                    return 0
                
                # Risk contribution of each asset
                risk_contrib = (weights * np.dot(cov_matrix, weights)) / portfolio_std
                target_contrib = 1.0 / len(weights)
                
                # Sum of squared deviations from equal risk contribution
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                          for _ in range(len(returns.columns)))
            
            n_assets = len(returns.columns)
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(risk_parity_objective, x0, method='SLSQP',
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                expected_returns = returns.mean() * 252
                portfolio_return = np.dot(weights, expected_returns)
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_std,
                    'sharpe_ratio': portfolio_return / portfolio_std if portfolio_std > 0 else 0,
                    'method': 'risk_parity',
                    'success': True
                }
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _optimize_equal_weight(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Equal weight portfolio"""
        try:
            n_assets = len(returns.columns)
            weights = np.array([1/n_assets] * n_assets)
            
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            return {
                'weights': dict(zip(returns.columns, weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': portfolio_return / portfolio_std if portfolio_std > 0 else 0,
                'method': 'equal_weight',
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _optimize_genetic_algorithm(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize using genetic algorithm"""
        try:
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_std == 0:
                    return -np.inf
                return -(portfolio_return / portfolio_std)  # Negative Sharpe ratio
            
            bounds = [(self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                     for _ in range(len(expected_returns))]
            
            result = differential_evolution(
                objective, bounds, 
                maxiter=self.config.max_iterations,
                popsize=self.config.population_size,
                seed=42
            )
            
            if result.success:
                weights = result.x
                weights = weights / np.sum(weights)  # Normalize
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_std,
                    'sharpe_ratio': portfolio_return / portfolio_std if portfolio_std > 0 else 0,
                    'method': 'genetic_algorithm',
                    'success': True
                }
            else:
                return {'success': False, 'error': 'Genetic algorithm failed to converge'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _optimize_mean_variance(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Mean-variance optimization"""
        try:
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Objective function (minimize variance)
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - 0.1}  # Target return
            ]
            bounds = tuple((self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                          for _ in range(len(expected_returns)))
            
            n_assets = len(expected_returns)
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(portfolio_variance, x0, method='SLSQP',
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_std = np.sqrt(portfolio_variance(weights))
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_std,
                    'sharpe_ratio': portfolio_return / portfolio_std if portfolio_std > 0 else 0,
                    'method': 'mean_variance',
                    'success': True
                }
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Portfolio returns
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Diversification metrics
            portfolio_std = np.sqrt(np.dot(list(weights.values()), 
                                         np.dot(returns.cov() * 252, list(weights.values()))))
            weighted_avg_std = sum(weights[asset] * returns[asset].std() * np.sqrt(252) 
                                 for asset in weights.keys())
            diversification_ratio = weighted_avg_std / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'diversification_ratio': diversification_ratio,
                'portfolio_std': portfolio_std,
                'weighted_avg_std': weighted_avg_std
            }
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def create_visualizations(self, data: pd.DataFrame, results: Dict[str, Any], 
                            output_dir: Path) -> List[Path]:
        """Create comprehensive visualizations"""
        if not self.config.generate_plots:
            return []
        
        plots = []
        
        try:
            # 1. Portfolio Weights Pie Chart
            if 'weights' in results and results['weights']:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                weights = results['weights']
                labels = list(weights.keys())
                sizes = list(weights.values())
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title('Portfolio Weights Distribution')
                plt.tight_layout()
                
                pie_path = output_dir / 'portfolio_weights.png'
                plt.savefig(pie_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(pie_path)
            
            # 2. Cumulative Returns Comparison
            if len(data.columns) > 1:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                # Individual assets
                for col in data.columns:
                    cumulative = (1 + data[col]).cumprod()
                    ax.plot(cumulative.index, cumulative.values, label=col, alpha=0.7)
                
                # Portfolio
                if 'weights' in results and results['weights']:
                    portfolio_returns = (data * pd.Series(results['weights'])).sum(axis=1)
                    portfolio_cumulative = (1 + portfolio_returns).cumprod()
                    ax.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                           label='Optimized Portfolio', linewidth=2, color='red')
                
                ax.set_title('Cumulative Returns Comparison')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                returns_path = output_dir / 'cumulative_returns.png'
                plt.savefig(returns_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(returns_path)
            
            # 3. Risk-Return Scatter Plot
            if len(data.columns) > 1:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                # Individual assets
                for col in data.columns:
                    annual_return = data[col].mean() * 252
                    annual_vol = data[col].std() * np.sqrt(252)
                    ax.scatter(annual_vol, annual_return, label=col, alpha=0.7, s=100)
                
                # Portfolio
                if 'expected_return' in results and 'volatility' in results:
                    ax.scatter(results['volatility'], results['expected_return'], 
                             label='Optimized Portfolio', s=200, color='red', marker='*')
                
                ax.set_xlabel('Volatility (Annualized)')
                ax.set_ylabel('Expected Return (Annualized)')
                ax.set_title('Risk-Return Profile')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                risk_return_path = output_dir / 'risk_return_profile.png'
                plt.savefig(risk_return_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(risk_return_path)
            
            # 4. Correlation Heatmap
            if len(data.columns) > 1:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                correlation_matrix = data.corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title('Asset Correlation Matrix')
                plt.tight_layout()
                
                corr_path = output_dir / 'correlation_matrix.png'
                plt.savefig(corr_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(corr_path)
            
            # 5. Rolling Sharpe Ratio
            if len(data.columns) > 0:
                fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
                
                window = 252  # 1 year
                for col in data.columns:
                    rolling_sharpe = data[col].rolling(window).mean() / data[col].rolling(window).std() * np.sqrt(252)
                    ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=col, alpha=0.7)
                
                ax.set_title(f'Rolling Sharpe Ratio ({window} days)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sharpe Ratio')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                sharpe_path = output_dir / 'rolling_sharpe.png'
                plt.savefig(sharpe_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots.append(sharpe_path)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
        
        return plots
    
    def run_single_backtest(self, data_file: Path, strategy_file: Path) -> List[Dict[str, Any]]:
        """Run single backtest with portfolio optimization"""
        try:
            # Load data
            data = self.load_data(data_file)
            if data is None or data.empty:
                return []
            
            # Load strategy
            strategy_cls = self.load_strategy(strategy_file)
            if strategy_cls is None:
                return []
            
            # Convert to returns
            returns = data.pct_change().dropna()
            
            # Optimize portfolio
            optimization_result = self.optimize_portfolio(returns)
            
            if not optimization_result.get('success', False):
                self.logger.warning(f"Portfolio optimization failed for {strategy_file.name}")
                return []
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(returns, optimization_result['weights'])
            
            # Create result
            result = {
                'data_file': data_file.name,
                'strategy_file': strategy_file.name,
                'engine': 'OptimizedPortfolioEngine',
                'timestamp': datetime.now().isoformat(),
                'optimization_method': optimization_result['method'],
                'portfolio_weights': optimization_result['weights'],
                'expected_return': optimization_result['expected_return'],
                'volatility': optimization_result['volatility'],
                'sharpe_ratio': optimization_result['sharpe_ratio'],
                **portfolio_metrics
            }
            
            # Create visualizations
            if self.config.generate_plots:
                output_dir = self.get_results_directory() / 'visualizations' / f"{data_file.stem}_{strategy_file.stem}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                plots = self.create_visualizations(data, optimization_result, output_dir)
                result['visualizations'] = [str(p) for p in plots]
            
            return [result]
            
        except Exception as e:
            self.logger.error(f"Error in portfolio backtest: {e}")
            return []
    
    def run(self):
        """Main execution method"""
        # Discover files
        data_files = self.discover_data_files()
        strategy_files = self.discover_strategy_files()
        
        if not data_files or not strategy_files:
            self.logger.error("No data files or strategy files found")
            return
        
        # Process combinations
        all_results = []
        total_combinations = len(data_files) * len(strategy_files)
        
        self.logger.info(f"Processing {total_combinations} portfolio optimization combinations...")
        
        for i, data_file in enumerate(data_files):
            for j, strategy_file in enumerate(strategy_files):
                combination_num = i * len(strategy_files) + j + 1
                self.logger.info(f"Processing combination {combination_num}/{total_combinations}: "
                               f"{data_file.name} + {strategy_file.name}")
                
                try:
                    results = self.run_single_backtest(data_file, strategy_file)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error processing {data_file.name} + {strategy_file.name}: {e}")
        
        # Save results
        self.save_results(all_results)
        
        # Final summary
        if all_results:
            df = pd.DataFrame(all_results)
            self.logger.info(f"Portfolio Optimization Summary:")
            self.logger.info(f"   • Total combinations: {len(all_results)}")
            self.logger.info(f"   • Strategies tested: {df['strategy_file'].nunique()}")
            self.logger.info(f"   • Data files tested: {df['data_file'].nunique()}")
            self.logger.info(f"   • Average Sharpe ratio: {df['sharpe_ratio'].mean():.2f}")
            self.logger.info(f"   • Best Sharpe ratio: {df['sharpe_ratio'].max():.2f}")
            self.logger.info(f"   • Average diversification ratio: {df['diversification_ratio'].mean():.2f}")
    
    def save_engine_specific_formats(self, result: Dict[str, Any], data_dir: Path):
        """Save portfolio-specific formats"""
        try:
            # Save portfolio weights as CSV
            if 'portfolio_weights' in result and result['portfolio_weights']:
                weights_df = pd.DataFrame(list(result['portfolio_weights'].items()), 
                                        columns=['Asset', 'Weight'])
                weights_path = data_dir / f"{data_dir.name}_portfolio_weights.csv"
                weights_df.to_csv(weights_path, index=False)
            
            # Save optimization details
            optimization_details = {
                'optimization_method': result.get('optimization_method', 'unknown'),
                'expected_return': result.get('expected_return', 0),
                'volatility': result.get('volatility', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'diversification_ratio': result.get('diversification_ratio', 0)
            }
            
            opt_path = data_dir / f"{data_dir.name}_optimization_details.json"
            with open(opt_path, 'w') as f:
                json.dump(optimization_details, f, indent=2)
            
            # Save performance metrics as CSV
            metrics_data = {
                'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 
                          'Max Drawdown', 'VaR 95%', 'CVaR 95%', 'Diversification Ratio'],
                'Value': [
                    result.get('total_return', 0),
                    result.get('annualized_return', 0),
                    result.get('volatility', 0),
                    result.get('sharpe_ratio', 0),
                    result.get('max_drawdown', 0),
                    result.get('var_95', 0),
                    result.get('cvar_95', 0),
                    result.get('diversification_ratio', 0)
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = data_dir / f"{data_dir.name}_performance_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio-specific formats: {e}")

def main():
    """Main entry point"""
    config = PortfolioEngineConfig()
    engine = OptimizedPortfolioEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
