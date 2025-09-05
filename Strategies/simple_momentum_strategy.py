#!/usr/bin/env python3
"""
Simple Momentum Strategy

A basic momentum-based trading strategy that:
- Buys when price momentum is positive
- Sells when price momentum is negative
- Uses simple moving averages for signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class SimpleMomentumStrategy:
    """Simple momentum-based trading strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize the strategy with parameters"""
        self.parameters = parameters or {
            'short_window': 10,
            'long_window': 30,
            'threshold': 0.02,
            'position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trading signals from price data"""
        try:
            if len(data) < self.parameters['long_window']:
                return pd.Series(0, index=data.index)
            
            # Calculate moving averages
            short_ma = data['close'].rolling(window=self.parameters['short_window']).mean()
            long_ma = data['close'].rolling(window=self.parameters['long_window']).mean()
            
            # Calculate momentum
            momentum = (short_ma - long_ma) / long_ma
            
            # Generate signals
            signals = pd.Series(0, index=data.index)
            
            # Buy signal: momentum above threshold
            buy_condition = momentum > self.parameters['threshold']
            signals[buy_condition] = 1
            
            # Sell signal: momentum below negative threshold
            sell_condition = momentum < -self.parameters['threshold']
            signals[sell_condition] = -1
            
            return signals
            
        except Exception as e:
            # Return neutral signals if there's an error
            return pd.Series(0, index=data.index)
    
    def backtest(self, data: pd.DataFrame, initial_cash: float = 100000.0, commission: float = 0.001) -> Dict[str, Any]:
        """Run backtest on the strategy"""
        try:
            if data is None or len(data) == 0:
                return {
                    'total_return': 0.0,
                    'final_cash': initial_cash,
                    'trades_count': 0,
                    'error': 'Empty data'
                }
            
            # Calculate signals
            signals = self.calculate_signals(data)
            
            # Initialize variables
            cash = initial_cash
            position = 0
            trades = []
            equity_curve = []
            
            # Simulate trading
            for i in range(len(data)):
                current_price = data['close'].iloc[i]
                signal = signals.iloc[i]
                
                # Calculate current equity
                current_equity = cash + (position * current_price)
                equity_curve.append(current_equity)
                
                # Execute trades based on signals
                if signal == 1 and position <= 0:  # Buy signal
                    if position < 0:
                        # Close short position
                        cash += abs(position) * current_price * (1 - commission)
                        trades.append({
                            'type': 'close_short',
                            'price': current_price,
                            'index': i
                        })
                    
                    # Open long position
                    position = int(cash * self.parameters['position_size'] / current_price)
                    if position > 0:
                        cash -= position * current_price * (1 + commission)
                        trades.append({
                            'type': 'buy',
                            'price': current_price,
                            'index': i
                        })
                
                elif signal == -1 and position >= 0:  # Sell signal
                    if position > 0:
                        # Close long position
                        cash += position * current_price * (1 - commission)
                        trades.append({
                            'type': 'sell',
                            'price': current_price,
                            'index': i
                        })
                        position = 0
                    
                    # Open short position
                    position = -int(cash * self.parameters['position_size'] / current_price)
                    if position < 0:
                        cash += abs(position) * current_price * (1 - commission)
                        trades.append({
                            'type': 'short',
                            'price': current_price,
                            'index': i
                        })
            
            # Close final position
            final_price = data['close'].iloc[-1]
            if position > 0:
                cash += position * final_price * (1 - commission)
            elif position < 0:
                cash -= abs(position) * final_price * (1 + commission)
            
            # Calculate returns
            total_return = (cash - initial_cash) / initial_cash
            
            return {
                'total_return': total_return,
                'final_cash': cash,
                'trades_count': len(trades),
                'equity_curve': equity_curve,
                'trades': trades
            }
            
        except Exception as e:
            return {
                'total_return': 0.0,
                'final_cash': initial_cash,
                'trades_count': 0,
                'error': str(e)
            }

# Strategy factory function
def create_strategy(parameters: Dict[str, Any] = None) -> SimpleMomentumStrategy:
    """Create a strategy instance with given parameters"""
    return SimpleMomentumStrategy(parameters)

# Default strategy parameters
DEFAULT_PARAMETERS = {
    'short_window': 10,
    'long_window': 30,
    'threshold': 0.02,
    'position_size': 0.1,
    'stop_loss': 0.05,
    'take_profit': 0.10
}

if __name__ == "__main__":
    # Test the strategy
    print("Simple Momentum Strategy loaded successfully!")
    print(f"Default parameters: {DEFAULT_PARAMETERS}")
