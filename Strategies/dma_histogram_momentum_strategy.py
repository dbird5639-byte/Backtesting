#!/usr/bin/env python3
"""
DMA Histogram Momentum Strategy

Based on Sunny Harris's methodology:
- Uses Dynamic Moving Average (DMA) histogram for early warning signals
- Monitors gold vs purple line dominance for momentum shifts
- Enters positions when histogram crosses zero line
- Uses Sunny Bands for entry/exit timing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class DMAHistogramMomentumStrategy:
    """DMA Histogram Momentum trading strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize the strategy with parameters"""
        self.parameters = parameters or {
            'dma_period': 20,
            'atr_period': 14,
            'band_multiplier': 2.0,
            'position_size': 0.1,
            'stop_loss_atr': 1.5,
            'histogram_threshold': 0.001,  # Minimum histogram change for signal
            'momentum_confirmation': 3  # Bars to confirm momentum
        }
    
    def calculate_dma(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Dynamic Moving Average"""
        close = data['close']
        dma = close.ewm(span=self.parameters['dma_period']).mean()
        return dma
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.parameters['atr_period']).mean()
        return atr
    
    def calculate_sunny_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Sunny Bands based on DMA and ATR"""
        dma = self.calculate_dma(data)
        atr = self.calculate_atr(data)
        
        # Calculate bands at average 2 ranges from DMA
        upper_inner = dma + (atr * self.parameters['band_multiplier'])
        upper_outer = dma + (atr * self.parameters['band_multiplier'] * 2)
        lower_inner = dma - (atr * self.parameters['band_multiplier'])
        lower_outer = dma - (atr * self.parameters['band_multiplier'] * 2)
        
        return {
            'dma': dma,
            'upper_inner': upper_inner,
            'upper_outer': upper_outer,
            'lower_inner': lower_inner,
            'lower_outer': lower_outer,
            'atr': atr
        }
    
    def calculate_dma_histogram(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate DMA Histogram for momentum signals"""
        dma = self.calculate_dma(data)
        
        # Calculate two different DMAs for histogram
        dma_fast = data['close'].ewm(span=self.parameters['dma_period']//2).mean()
        dma_slow = data['close'].ewm(span=self.parameters['dma_period']).mean()
        
        # Histogram is the difference between fast and slow DMA
        histogram = dma_fast - dma_slow
        
        # Determine which line is on top (gold vs purple)
        gold_on_top = dma_fast > dma_slow
        purple_on_top = dma_slow > dma_fast
        
        # Calculate histogram momentum (rate of change)
        histogram_momentum = histogram.diff()
        
        return {
            'histogram': histogram,
            'histogram_momentum': histogram_momentum,
            'gold_on_top': gold_on_top,
            'purple_on_top': purple_on_top,
            'dma_fast': dma_fast,
            'dma_slow': dma_slow
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trading signals from price data"""
        try:
            if len(data) < self.parameters['dma_period'] + self.parameters['atr_period']:
                return pd.Series(0, index=data.index)
            
            # Calculate DMA Histogram and Sunny Bands
            histogram_data = self.calculate_dma_histogram(data)
            bands = self.calculate_sunny_bands(data)
            
            # Initialize signals
            signals = pd.Series(0, index=data.index)
            
            for i in range(2, len(data)):
                current_histogram = histogram_data['histogram'].iloc[i]
                prev_histogram = histogram_data['histogram'].iloc[i-1]
                prev2_histogram = histogram_data['histogram'].iloc[i-2]
                
                current_momentum = histogram_data['histogram_momentum'].iloc[i]
                gold_top = histogram_data['gold_on_top'].iloc[i]
                purple_top = histogram_data['purple_on_top'].iloc[i]
                
                # Check for momentum shift signals
                # Bearish momentum: gold on top, histogram decreasing, crossing zero
                if (gold_top and 
                    current_momentum < -self.parameters['histogram_threshold'] and
                    prev_histogram > 0 and current_histogram < 0):
                    
                    # Confirm with Sunny Bands for short entry
                    current_price = data['close'].iloc[i]
                    if current_price >= bands['upper_inner'].iloc[i]:
                        signals.iloc[i] = -1  # SHORT signal
                
                # Bullish momentum: purple on top, histogram increasing, crossing zero
                elif (purple_top and 
                      current_momentum > self.parameters['histogram_threshold'] and
                      prev_histogram < 0 and current_histogram > 0):
                    
                    # Confirm with Sunny Bands for long entry
                    current_price = data['close'].iloc[i]
                    if current_price <= bands['lower_inner'].iloc[i]:
                        signals.iloc[i] = 1  # LONG signal
                
                # Additional momentum confirmation signals
                # Strong bearish momentum: histogram decreasing while gold on top
                elif (gold_top and 
                      current_momentum < -self.parameters['histogram_threshold'] and
                      abs(current_histogram) > abs(prev_histogram) and
                      current_histogram < 0):
                    
                    current_price = data['close'].iloc[i]
                    if current_price >= bands['upper_inner'].iloc[i]:
                        signals.iloc[i] = -1  # SHORT signal
                
                # Strong bullish momentum: histogram increasing while purple on top
                elif (purple_top and 
                      current_momentum > self.parameters['histogram_threshold'] and
                      abs(current_histogram) > abs(prev_histogram) and
                      current_histogram > 0):
                    
                    current_price = data['close'].iloc[i]
                    if current_price <= bands['lower_inner'].iloc[i]:
                        signals.iloc[i] = 1  # LONG signal
            
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
            entry_price = 0
            entry_index = 0
            
            # Simulate trading
            for i in range(len(data)):
                current_price = data['close'].iloc[i]
                signal = signals.iloc[i]
                
                # Calculate current equity
                current_equity = cash + (position * current_price)
                equity_curve.append(current_equity)
                
                # Execute trades based on signals
                if signal == 1 and position <= 0:  # Long signal
                    if position < 0:
                        # Close short position
                        cash += abs(position) * current_price * (1 - commission)
                        trades.append({
                            'type': 'close_short',
                            'price': current_price,
                            'index': i,
                            'profit': (entry_price - current_price) * abs(position)
                        })
                    
                    # Open long position
                    position = int(cash * self.parameters['position_size'] / current_price)
                    if position > 0:
                        cash -= position * current_price * (1 + commission)
                        entry_price = current_price
                        entry_index = i
                        trades.append({
                            'type': 'open_long',
                            'price': current_price,
                            'index': i,
                            'position_size': position
                        })
                
                elif signal == -1 and position >= 0:  # Short signal
                    if position > 0:
                        # Close long position
                        cash += position * current_price * (1 - commission)
                        trades.append({
                            'type': 'close_long',
                            'price': current_price,
                            'index': i,
                            'profit': (current_price - entry_price) * position
                        })
                    
                    # Open short position
                    position = -int(cash * self.parameters['position_size'] / current_price)
                    if position < 0:
                        cash += abs(position) * current_price * (1 - commission)
                        entry_price = current_price
                        entry_index = i
                        trades.append({
                            'type': 'open_short',
                            'price': current_price,
                            'index': i,
                            'position_size': abs(position)
                        })
                
                # Check stop loss and take profit
                if position != 0:
                    bands = self.calculate_sunny_bands(data.iloc[:i+1])
                    atr = bands['atr'].iloc[-1]
                    
                    # Stop loss based on ATR
                    stop_loss = self.parameters['stop_loss_atr'] * atr
                    
                    if position > 0:  # Long position
                        if current_price <= entry_price - stop_loss:
                            # Stop loss hit
                            cash += position * current_price * (1 - commission)
                            trades.append({
                                'type': 'stop_loss_long',
                                'price': current_price,
                                'index': i,
                                'loss': (entry_price - current_price) * position
                            })
                            position = 0
                        elif current_price >= bands['upper_inner'].iloc[i]:
                            # Take profit at upper inner band
                            cash += position * current_price * (1 - commission)
                            trades.append({
                                'type': 'take_profit_long',
                                'price': current_price,
                                'index': i,
                                'profit': (current_price - entry_price) * position
                            })
                            position = 0
                    
                    elif position < 0:  # Short position
                        if current_price >= entry_price + stop_loss:
                            # Stop loss hit
                            cash += abs(position) * current_price * (1 - commission)
                            trades.append({
                                'type': 'stop_loss_short',
                                'price': current_price,
                                'index': i,
                                'loss': (current_price - entry_price) * abs(position)
                            })
                            position = 0
                        elif current_price <= bands['lower_inner'].iloc[i]:
                            # Take profit at lower inner band
                            cash += abs(position) * current_price * (1 - commission)
                            trades.append({
                                'type': 'take_profit_short',
                                'price': current_price,
                                'index': i,
                                'profit': (entry_price - current_price) * abs(position)
                            })
                            position = 0
            
            # Close any remaining position
            if position != 0:
                final_price = data['close'].iloc[-1]
                if position > 0:
                    cash += position * final_price * (1 - commission)
                else:
                    cash += abs(position) * final_price * (1 - commission)
            
            # Calculate final results
            total_return = (cash - initial_cash) / initial_cash
            final_cash = cash
            
            return {
                'total_return': total_return,
                'final_cash': final_cash,
                'trades_count': len(trades),
                'trades': trades,
                'equity_curve': equity_curve,
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'sharpe_ratio': self._calculate_sharpe_ratio(equity_curve),
                'win_rate': self._calculate_win_rate(trades)
            }
            
        except Exception as e:
            return {
                'total_return': 0.0,
                'final_cash': initial_cash,
                'trades_count': 0,
                'error': str(e)
            }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        return returns.mean() / returns.std() if returns.std() != 0 else 0.0
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        profitable_trades = [t for t in trades if 'profit' in t and t['profit'] > 0]
        return len(profitable_trades) / len(trades) if trades else 0.0
