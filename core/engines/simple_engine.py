"""
Simple Backtesting Engine

A basic implementation of the backtesting engine that provides
fundamental backtesting functionality without advanced features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
import time

from ..base import BaseEngine, EngineConfig, BacktestResult, BaseStrategy
from ..data import BaseDataHandler
from ..risk_management import BaseRiskManager


class SimpleEngine(BaseEngine):
    """
    Simple backtesting engine implementation.
    
    This engine provides basic backtesting functionality:
    - Single strategy execution
    - Basic performance metrics
    - Simple result handling
    - No advanced features like ML or portfolio optimization
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize the simple engine.
        
        Args:
            config: Engine configuration
        """
        super().__init__(config)
        self.logger.info("Initialized Simple Backtesting Engine")
    
    def load_data(self, symbol: str, timeframe: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load market data for backtesting.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info(f"Loading data for {symbol} {timeframe}")
        
        # For now, we'll create dummy data
        # In a real implementation, this would use the data handler
        if self.data_handler:
            return self.data_handler.load_data(symbol, timeframe, start_date, end_date)
        
        # Create dummy data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        self.logger.info(f"Loaded {len(data)} data points for {symbol} {timeframe}")
        return data
    
    def load_strategy(self, strategy_name: str, **parameters) -> BaseStrategy:
        """
        Load a trading strategy.
        
        Args:
            strategy_name: Name of the strategy to load
            **parameters: Strategy parameters
            
        Returns:
            Loaded strategy instance
        """
        self.logger.info(f"Loading strategy: {strategy_name}")
        
        # For now, we'll create a dummy strategy
        # In a real implementation, this would dynamically load strategy classes
        from ..strategies import BaseStrategy as StrategyBase
        
        # Create a simple moving average strategy
        class SimpleMAStrategy(StrategyBase):
            def __init__(self, config):
                super().__init__(config)
                self.short_window = parameters.get('short_window', 10)
                self.long_window = parameters.get('long_window', 20)
            
            def initialize(self, data):
                return len(data) >= self.long_window
            
            def generate_signals(self, data):
                if len(data) < self.long_window:
                    return []
                
                # Calculate moving averages
                short_ma = data['Close'].rolling(window=self.short_window).mean()
                long_ma = data['Close'].rolling(window=self.long_window).mean()
                
                signals = []
                current_price = data['Close'].iloc[-1]
                
                # Generate buy signal when short MA crosses above long MA
                if (short_ma.iloc[-1] > long_ma.iloc[-1] and 
                    short_ma.iloc[-2] <= long_ma.iloc[-2]):
                    signals.append(self.Signal(
                        timestamp=data.index[-1],
                        signal_type='buy',
                        symbol=self.config.symbol,
                        price=current_price,
                        quantity=1.0,
                        reason='Golden cross'
                    ))
                
                # Generate sell signal when short MA crosses below long MA
                elif (short_ma.iloc[-1] < long_ma.iloc[-1] and 
                      short_ma.iloc[-2] >= long_ma.iloc[-2]):
                    signals.append(self.Signal(
                        timestamp=data.index[-1],
                        signal_type='sell',
                        symbol=self.config.symbol,
                        price=current_price,
                        quantity=1.0,
                        reason='Death cross'
                    ))
                
                return signals
        
        # Create strategy config
        from ..base import StrategyConfig
        strategy_config = StrategyConfig(
            name=strategy_name,
            symbol=parameters.get('symbol', 'BTC'),
            timeframe=parameters.get('timeframe', '1h')
        )
        
        strategy = SimpleMAStrategy(strategy_config)
        self.strategies.append(strategy)
        
        self.logger.info(f"Strategy {strategy_name} loaded successfully")
        return strategy
    
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
        start_time = time.time()
        self.logger.info(f"Starting backtest for strategy: {strategy.config.name}")
        
        # Initialize strategy
        if not strategy.initialize(data):
            raise ValueError("Strategy initialization failed")
        
        # Initialize backtest variables
        initial_cash = kwargs.get('initial_cash', self.config.initial_cash)
        current_cash = initial_cash
        position = 0
        trades = []
        equity_curve = []
        
        # Run backtest
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            if len(current_data) < 2:  # Need at least 2 data points for signals
                continue
            
            # Get signals from strategy
            signals = strategy.generate_signals(current_data)
            
            current_price = current_data['Close'].iloc[-1]
            current_timestamp = current_data.index[-1]
            
            # Process signals
            for signal in signals:
                if signal.signal_type == 'buy' and position == 0:
                    # Execute buy order
                    shares = current_cash / current_price
                    position = shares
                    current_cash = 0
                    
                    trades.append({
                        'timestamp': current_timestamp,
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price
                    })
                    
                elif signal.signal_type == 'sell' and position > 0:
                    # Execute sell order
                    current_cash = position * current_price
                    position = 0
                    
                    trades.append({
                        'timestamp': current_timestamp,
                        'type': 'sell',
                        'price': current_price,
                        'shares': position,
                        'value': position * current_price
                    })
            
            # Calculate current portfolio value
            portfolio_value = current_cash + (position * current_price)
            equity_curve.append(portfolio_value)
        
        # Calculate final portfolio value
        final_value = current_cash + (position * data['Close'].iloc[-1])
        total_return = (final_value - initial_cash) / initial_cash
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(equity_series)
        
        # Trade metrics
        if trades:
            winning_trades = [t for t in trades if t['type'] == 'sell' and t['value'] > 0]
            win_rate = len(winning_trades) / len([t for t in trades if t['type'] == 'sell'])
        else:
            win_rate = 0.0
        
        execution_time = time.time() - start_time
        
        # Create result object
        result = BacktestResult(
            strategy_name=strategy.config.name,
            symbol=strategy.config.symbol,
            timeframe=strategy.config.timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_return=total_return,
            annualized_return=total_return * (252 / len(data)),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio,  # Simplified
            calmar_ratio=total_return / max_drawdown if max_drawdown > 0 else 0,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=1.0,  # Simplified
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t['type'] == 'sell' and t['value'] > 0]),
            losing_trades=len([t for t in trades if t['type'] == 'sell' and t['value'] <= 0]),
            avg_win=0.0,  # Simplified
            avg_loss=0.0,  # Simplified
            largest_win=0.0,  # Simplified
            largest_loss=0.0,  # Simplified
            volatility=returns.std() * np.sqrt(252),
            var_95=self._calculate_var(returns, 0.95),
            cvar_95=self._calculate_cvar(returns, 0.95),
            equity_curve=equity_series,
            trade_log=trades,
            execution_time=execution_time
        )
        
        self.results.append(result)
        self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        self.logger.info(f"Total return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}")
        
        return result
    
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
        self.logger.info(f"Starting portfolio backtest with {len(strategies)} strategies")
        
        results = []
        for strategy in strategies:
            symbol = strategy.config.symbol
            if symbol in data_dict:
                result = self.run_backtest(strategy, data_dict[symbol], **kwargs)
                results.append(result)
            else:
                self.logger.warning(f"No data found for symbol: {symbol}")
        
        return results
    
    def optimize_strategy(self, strategy: BaseStrategy, data: pd.DataFrame, 
                         param_ranges: Dict[str, List[Any]], **kwargs) -> tuple:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy: Strategy to optimize
            data: Market data for optimization
            param_ranges: Parameter ranges to test
            **kwargs: Additional optimization parameters
            
        Returns:
            Tuple of (best_parameters, best_result)
        """
        self.logger.info(f"Starting parameter optimization for {strategy.config.name}")
        
        best_result = None
        best_params = None
        best_sharpe = float('-inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations:
            # Update strategy parameters
            strategy.update_parameters(params)
            
            # Run backtest with current parameters
            try:
                result = self.run_backtest(strategy, data, **kwargs)
                
                # Check if this is the best result so far
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_result = result
                    best_params = params.copy()
                
            except Exception as e:
                self.logger.warning(f"Backtest failed for parameters {params}: {e}")
                continue
        
        if best_result is None:
            raise ValueError("No successful backtests during optimization")
        
        self.logger.info(f"Optimization completed. Best Sharpe: {best_sharpe:.2f}")
        return best_params, best_result
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for grid search"""
        import itertools
        
        keys = param_ranges.keys()
        values = param_ranges.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
