#!/usr/bin/env python3
"""
Multi-Timeframe Momentum Strategy

A sophisticated strategy that combines momentum signals from multiple timeframes:
- 1H, 4H, and 1D timeframe analysis
- Momentum convergence for high-probability entries
- Dynamic position sizing based on signal strength
- Advanced risk management with trailing stops
"""

import pandas as pd
import numpy as np
import talib
import logging
from backtesting import Strategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTimeframeMomentumStrategy(Strategy):
    """
    Multi-Timeframe Momentum Strategy
    
    Combines momentum signals from multiple timeframes:
    - 1H: Short-term momentum and entry timing
    - 4H: Medium-term trend confirmation
    - 1D: Long-term trend direction
    - Only trades when all timeframes align
    """
    
    # Core parameters
    risk_per_trade = 0.01          # 1% risk per trade
    max_positions = 2              # Maximum concurrent positions
    max_drawdown = 0.10            # Maximum drawdown limit (10%)
    consecutive_loss_limit = 3      # Maximum consecutive losses
    
    # Multi-timeframe parameters
    tf_1h = 60                     # 1-hour timeframe (in minutes)
    tf_4h = 240                    # 4-hour timeframe (in minutes)
    tf_1d = 1440                   # 1-day timeframe (in minutes)
    
    # Momentum indicators
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
    # MACD parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Moving averages
    ema_fast = 20
    ema_medium = 50
    ema_slow = 200
    
    # Volatility
    atr_period = 14
    atr_multiplier = 2.0
    
    # Volume analysis
    volume_period = 20
    volume_threshold = 1.3
    
    # Signal strength thresholds
    min_signal_strength = 0.7      # Minimum combined signal strength
    momentum_threshold = 0.6       # Minimum momentum alignment
    
    # Performance tracking
    track_performance = True
    performance_window = 100
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.entry_prices = {}
        self.stop_losses = {}
        self.take_profits = {}
        
        # Core indicators (current timeframe)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.data.Close,
                                                            fastperiod=self.macd_fast,
                                                            slowperiod=self.macd_slow,
                                                            signalperiod=self.macd_signal)
        
        # Moving averages
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema_50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_medium)
        self.ema_200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        
        # Volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # Volume
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
        # Additional momentum indicators
        self.stoch_k, self.stoch_d = self.I(talib.STOCH, self.data.High, self.data.Low, 
                                           self.data.Close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Williams %R for momentum confirmation
        self.williams_r = self.I(talib.WILLR, self.data.High, self.data.Low, 
                                self.data.Close, timeperiod=14)
        
        # CCI for trend strength
        self.cci = self.I(talib.CCI, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=14)
        
        logger.info("MultiTimeframeMomentumStrategy initialized successfully")
    
    def calculate_multi_timeframe_signals(self) -> dict:
        """Calculate momentum signals for multiple timeframes"""
        if len(self.data) < max(self.ema_slow, self.rsi_period * 2):
            return {}
        
        # Current timeframe signals (1H equivalent)
        current_signals = self.calculate_timeframe_signals(self.data, 1)
        
        # Higher timeframe signals (4H equivalent)
        tf_4h_data = self.resample_data(self.data, self.tf_4h)
        tf_4h_signals = self.calculate_timeframe_signals(tf_4h_data, 4)
        
        # Daily timeframe signals
        tf_1d_data = self.resample_data(self.data, self.tf_1d)
        tf_1d_signals = self.calculate_timeframe_signals(tf_1d_data, 24)
        
        return {
            '1h': current_signals,
            '4h': tf_4h_signals,
            '1d': tf_1d_signals
        }
    
    def resample_data(self, data, timeframe_minutes: int) -> pd.DataFrame:
        """Resample data to higher timeframe"""
        # This is a simplified resampling - in real implementation, you'd use proper OHLCV resampling
        # For now, we'll use every nth bar to simulate higher timeframes
        
        if timeframe_minutes <= 1:
            return data
        
        # Calculate how many bars to skip
        bars_to_skip = max(1, timeframe_minutes // 1)  # Assuming 1-minute bars
        
        # Create resampled data
        resampled_data = {
            'Open': data.Open[::bars_to_skip],
            'High': data.High[::bars_to_skip],
            'Low': data.Low[::bars_to_skip],
            'Close': data.Close[::bars_to_skip],
            'Volume': data.Volume[::bars_to_skip]
        }
        
        return pd.DataFrame(resampled_data)
    
    def calculate_timeframe_signals(self, data: pd.DataFrame, timeframe_hours: int) -> dict:
        """Calculate momentum signals for a specific timeframe"""
        if len(data) < 50:
            return {}
        
        # RSI momentum
        rsi = talib.RSI(data.Close, timeperiod=self.rsi_period)
        rsi_momentum = 0.0
        if len(rsi) > 0:
            current_rsi = rsi[-1]
            if current_rsi < 30:
                rsi_momentum = 1.0  # Strong oversold
            elif current_rsi < 40:
                rsi_momentum = 0.7  # Moderately oversold
            elif current_rsi > 70:
                rsi_momentum = -1.0  # Strong overbought
            elif current_rsi > 60:
                rsi_momentum = -0.7  # Moderately overbought
            else:
                rsi_momentum = 0.0  # Neutral
        
        # MACD momentum
        macd, macd_signal, _ = talib.MACD(data.Close, fastperiod=self.macd_fast,
                                         slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        macd_momentum = 0.0
        if len(macd) > 1 and len(macd_signal) > 1:
            if macd[-1] > macd_signal[-1] and macd[-1] > macd[-2]:
                macd_momentum = 1.0  # Strong bullish momentum
            elif macd[-1] > macd_signal[-1]:
                macd_momentum = 0.5  # Moderate bullish momentum
            elif macd[-1] < macd_signal[-1] and macd[-1] < macd[-2]:
                macd_momentum = -1.0  # Strong bearish momentum
            elif macd[-1] < macd_signal[-1]:
                macd_momentum = -0.5  # Moderate bearish momentum
        
        # Moving average trend
        ema_20 = talib.EMA(data.Close, timeperiod=self.ema_fast)
        ema_50 = talib.EMA(data.Close, timeperiod=self.ema_medium)
        ema_200 = talib.EMA(data.Close, timeperiod=self.ema_slow)
        
        trend_momentum = 0.0
        if len(ema_20) > 0 and len(ema_50) > 0 and len(ema_200) > 0:
            if ema_20[-1] > ema_50[-1] > ema_200[-1]:
                trend_momentum = 1.0  # Strong uptrend
            elif ema_20[-1] > ema_50[-1]:
                trend_momentum = 0.5  # Moderate uptrend
            elif ema_20[-1] < ema_50[-1] < ema_200[-1]:
                trend_momentum = -1.0  # Strong downtrend
            elif ema_20[-1] < ema_50[-1]:
                trend_momentum = -0.5  # Moderate downtrend
        
        # Volume momentum
        volume_ma = talib.SMA(data.Volume, timeperiod=self.volume_period)
        volume_momentum = 0.0
        if len(volume_ma) > 0 and len(data.Volume) > 0:
            if data.Volume[-1] > volume_ma[-1] * 1.5:
                volume_momentum = 1.0  # High volume
            elif data.Volume[-1] > volume_ma[-1]:
                volume_momentum = 0.5  # Above average volume
            else:
                volume_momentum = 0.0  # Below average volume
        
        # Calculate combined momentum score
        momentum_score = (rsi_momentum * 0.3 + macd_momentum * 0.3 + 
                         trend_momentum * 0.25 + volume_momentum * 0.15)
        
        return {
            'rsi_momentum': rsi_momentum,
            'macd_momentum': macd_momentum,
            'trend_momentum': trend_momentum,
            'volume_momentum': volume_momentum,
            'combined_momentum': momentum_score,
            'timeframe': timeframe_hours
        }
    
    def should_enter_long(self) -> bool:
        """Determine if we should enter a long position based on multi-timeframe signals"""
        if len(self.data) < max(self.ema_slow, self.rsi_period * 2):
            return False
        
        # Get multi-timeframe signals
        signals = self.calculate_multi_timeframe_signals()
        if not signals:
            return False
        
        # Check if all timeframes show bullish momentum
        long_signals = []
        for tf, tf_signals in signals.items():
            if tf_signals.get('combined_momentum', 0) > self.momentum_threshold:
                long_signals.append(tf_signals['combined_momentum'])
        
        # Need at least 2 timeframes showing bullish momentum
        if len(long_signals) < 2:
            return False
        
        # Calculate overall signal strength
        signal_strength = np.mean(long_signals)
        if signal_strength < self.min_signal_strength:
            return False
        
        # Additional confirmation from current timeframe
        current_price = self.data.Close[-1]
        price_above_ema = current_price > self.ema_20[-1]
        rsi_not_overbought = self.rsi[-1] < self.rsi_overbought
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        return price_above_ema and rsi_not_overbought and volume_above_average
    
    def should_enter_short(self) -> bool:
        """Determine if we should enter a short position based on multi-timeframe signals"""
        if len(self.data) < max(self.ema_slow, self.rsi_period * 2):
            return False
        
        # Get multi-timeframe signals
        signals = self.calculate_multi_timeframe_signals()
        if not signals:
            return False
        
        # Check if all timeframes show bearish momentum
        short_signals = []
        for tf, tf_signals in signals.items():
            if tf_signals.get('combined_momentum', 0) < -self.momentum_threshold:
                short_signals.append(abs(tf_signals['combined_momentum']))
        
        # Need at least 2 timeframes showing bearish momentum
        if len(short_signals) < 2:
            return False
        
        # Calculate overall signal strength
        signal_strength = np.mean(short_signals)
        if signal_strength < self.min_signal_strength:
            return False
        
        # Additional confirmation from current timeframe
        current_price = self.data.Close[-1]
        price_below_ema = current_price < self.ema_20[-1]
        rsi_not_oversold = self.rsi[-1] > self.rsi_oversold
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        return price_below_ema and rsi_not_oversold and volume_above_average
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion and signal strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Get signal strength for position sizing
        signals = self.calculate_multi_timeframe_signals()
        signal_strength = 0.0
        
        if signals:
            # Calculate average signal strength across timeframes
            momentum_scores = [s.get('combined_momentum', 0) for s in signals.values()]
            signal_strength = np.mean([abs(score) for score in momentum_scores])
        
        # Calculate base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust position size based on signal strength
        if signal_strength > 0.8:
            position_multiplier = 1.2  # Strong signals get larger positions
        elif signal_strength > 0.6:
            position_multiplier = 1.0  # Moderate signals get normal positions
        else:
            position_multiplier = 0.7  # Weak signals get smaller positions
        
        adjusted_position_size = base_position_size * position_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss based on volatility and support/resistance"""
        atr_value = self.atr[-1]
        
        if is_long:
            # For longs, look for support levels
            stop_loss = entry_price - (atr_value * self.atr_multiplier)
            
            # Additional support from moving averages
            if self.ema_50[-1] < entry_price:
                stop_loss = max(stop_loss, self.ema_50[-1] * 0.995)
        else:
            # For shorts, look for resistance levels
            stop_loss = entry_price + (atr_value * self.atr_multiplier)
            
            # Additional resistance from moving averages
            if self.ema_50[-1] > entry_price:
                stop_loss = min(stop_loss, self.ema_50[-1] * 1.005)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit based on risk-reward and volatility"""
        risk = abs(entry_price - stop_loss)
        atr_value = self.atr[-1]
        
        # Dynamic risk-reward based on volatility
        if atr_value > np.mean(self.atr[-20:]) * 1.5:
            # High volatility - tighter profit targets
            risk_reward_ratio = 2.0
        else:
            # Normal volatility - standard profit targets
            risk_reward_ratio = 2.5
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for new entry opportunities based on multi-timeframe signals"""
        current_price = self.data.Close[-1]
        
        # Long entry
        if self.should_enter_long():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG", current_price, stop_loss, take_profit, position_size)
        
        # Short entry
        elif self.should_enter_short():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with trailing stops and profit taking"""
        for position in self.positions:
            # Implement trailing stops for profitable positions
            if position.pl > 0:
                # Move stop loss to breakeven after 1:1 risk-reward
                entry_price = position.entry_price
                current_price = self.data.Close[-1]
                
                if position.is_long:
                    risk = entry_price - position.sl
                    if current_price > entry_price + risk:
                        # Move stop to breakeven
                        new_stop = entry_price + (risk * 0.1)  # Small buffer above breakeven
                        if new_stop > position.sl:
                            position.sl = new_stop
                else:
                    risk = position.sl - entry_price
                    if current_price < entry_price - risk:
                        # Move stop to breakeven
                        new_stop = entry_price - (risk * 0.1)  # Small buffer below breakeven
                        if new_stop < position.sl:
                            position.sl = new_stop
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        # Get signal strength at entry
        signals = self.calculate_multi_timeframe_signals()
        signal_strength = 0.0
        if signals:
            momentum_scores = [s.get('combined_momentum', 0) for s in signals.values()]
            signal_strength = np.mean([abs(score) for score in momentum_scores])
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'signal_strength': signal_strength,
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Signal Strength={signal_strength:.2f}")
    
    def on_trade_exit(self, position, exit_price, exit_time):
        """Handle trade exit events"""
        # Update consecutive losses
        if position.pl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Log trade result
        logger.info(f"Trade exited: P&L={position.pl:.4f}, "
                   f"Consecutive losses={self.consecutive_losses}")
        
        # Check if we should reduce position sizes after consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            logger.warning(f"Consecutive losses limit reached: {self.consecutive_losses}")
            # Could implement position size reduction logic here
