#!/usr/bin/env python3
"""
Technical Breakout Cascade Strategy

A sophisticated strategy designed to capitalize on technical breakout cascades during bull runs:
- Detects technical breakout patterns and cascade effects
- Identifies chain reaction breakouts across multiple levels
- Uses multi-timeframe analysis for breakout confirmation
- Implements position sizing based on breakout strength
- Focuses on capturing the most profitable technical breakouts

This strategy targets the most influential technical breakouts during bull run conditions.
"""

import pandas as pd
import numpy as np
import talib
import logging
from backtesting import Strategy
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BreakoutPhase(Enum):
    """Breakout phases for strategic positioning"""
    COMPRESSION = "compression"        # Price compression before breakout
    INITIAL_BREAKOUT = "initial_breakout"  # Initial breakout
    CASCADE = "cascade"                # Breakout cascade
    ACCELERATION = "acceleration"      # Breakout acceleration
    EXHAUSTION = "exhaustion"          # Breakout exhaustion

class TechnicalBreakoutCascadeStrategy(Strategy):
    """
    Technical Breakout Cascade Strategy
    
    Captures technical breakout cascades during bull runs:
    - Detects technical breakout patterns and cascade effects
    - Identifies chain reaction breakouts across multiple levels
    - Uses multi-timeframe analysis for breakout confirmation
    - Implements position sizing based on breakout strength
    """
    
    # Core parameters
    risk_per_trade = 0.017             # 1.7% risk per trade
    max_positions = 4                  # Maximum concurrent positions
    max_drawdown = 0.13                # Maximum drawdown limit
    consecutive_loss_limit = 3         # Maximum consecutive losses
    
    # Breakout detection parameters
    breakout_lookback = 30             # Periods to analyze for breakout patterns
    breakout_threshold = 0.02          # Breakout threshold
    cascade_threshold = 0.8            # Cascade threshold
    acceleration_threshold = 0.6       # Acceleration threshold
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 35
    rsi_overbought = 65
    atr_period = 14
    atr_multiplier = 2.5
    
    # MACD parameters
    macd_fast = 8
    macd_slow = 21
    macd_signal = 5
    
    # Volume analysis
    volume_period = 20
    volume_breakout_threshold = 2.0
    
    # Breakout indicators
    breakout_period = 20
    cascade_period = 10
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.breakout_phases = []
        self.cascade_events = []
        
        # Technical indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # MACD for momentum
        self.macd, self.macd_signal_line, self.macd_histogram = self.I(
            talib.MACD, self.data.Close, 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
        # Bollinger Bands for volatility
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Stochastic for momentum
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, self.data.High, self.data.Low, self.data.Close,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R for momentum
        self.williams_r = self.I(talib.WILLR, self.data.High, self.data.Low, 
                                self.data.Close, timeperiod=14)
        
        # Breakout indicators
        self.breakout_strength = self.I(self._calculate_breakout_strength)
        self.cascade_signal = self.I(self._calculate_cascade_signal)
        self.acceleration_signal = self.I(self._calculate_acceleration_signal)
        self.compression_signal = self.I(self._calculate_compression_signal)
        self.breakout_momentum = self.I(self._calculate_breakout_momentum)
    
    def _calculate_breakout_strength(self, data):
        """Calculate breakout strength indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Price momentum
        price_momentum = data.Close.pct_change(3)
        
        # Volume surge
        volume_surge = data.Volume / self.volume_ma
        
        # RSI momentum
        rsi_momentum = self.rsi.diff(2)
        
        # MACD momentum
        macd_momentum = self.macd_histogram.diff(1)
        
        # Combine breakout factors
        breakout_strength = (
            abs(price_momentum) * 0.4 +
            volume_surge * 0.3 +
            abs(rsi_momentum) * 0.2 +
            abs(macd_momentum) * 0.1
        )
        
        return breakout_strength.rolling(window=self.breakout_period).mean()
    
    def _calculate_cascade_signal(self, data):
        """Calculate cascade signal indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Multiple breakout levels
        breakout_1 = data.Close > data.Close.rolling(20).max().shift(1)
        breakout_2 = data.Close > data.Close.rolling(50).max().shift(1)
        breakout_3 = data.Close > data.Close.rolling(100).max().shift(1)
        
        # Volume confirmation
        volume_strong = data.Volume > self.volume_ma * 1.5
        
        # Cascade signal
        cascade_signal = (
            breakout_1.astype(int) * 0.3 +
            breakout_2.astype(int) * 0.3 +
            breakout_3.astype(int) * 0.2 +
            volume_strong.astype(int) * 0.2
        )
        
        return cascade_signal.rolling(window=self.cascade_period).mean()
    
    def _calculate_acceleration_signal(self, data):
        """Calculate acceleration signal indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Price acceleration
        price_acceleration = data.Close.pct_change().diff()
        
        # Volume acceleration
        volume_acceleration = data.Volume.pct_change().diff()
        
        # RSI acceleration
        rsi_acceleration = self.rsi.diff().diff()
        
        # Combine acceleration factors
        acceleration_signal = (
            abs(price_acceleration) * 0.5 +
            abs(volume_acceleration) * 0.3 +
            abs(rsi_acceleration) * 0.2
        )
        
        return acceleration_signal.rolling(window=5).mean()
    
    def _calculate_compression_signal(self, data):
        """Calculate compression signal indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Bollinger Band compression
        bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        bb_compression = bb_width < bb_width.rolling(20).mean() * 0.8
        
        # ATR compression
        atr_compression = self.atr < self.atr.rolling(20).mean() * 0.8
        
        # Volume compression
        volume_compression = data.Volume < self.volume_ma * 0.8
        
        # Combine compression factors
        compression_signal = (
            bb_compression.astype(int) * 0.4 +
            atr_compression.astype(int) * 0.3 +
            volume_compression.astype(int) * 0.3
        )
        
        return compression_signal.rolling(window=5).mean()
    
    def _calculate_breakout_momentum(self, data):
        """Calculate breakout momentum indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Breakout momentum based on strength and cascade
        strength_momentum = self.breakout_strength.diff(3)
        cascade_momentum = self.cascade_signal.diff(3)
        acceleration_momentum = self.acceleration_signal.diff(3)
        
        # Combine momentum factors
        breakout_momentum = (
            strength_momentum * 0.4 +
            cascade_momentum * 0.3 +
            acceleration_momentum * 0.3
        )
        
        return breakout_momentum.rolling(window=5).mean()
    
    def detect_breakout_phase(self) -> Tuple[BreakoutPhase, float]:
        """Detect the current breakout phase"""
        if len(self.data) < self.breakout_lookback:
            return BreakoutPhase.COMPRESSION, 0.5
        
        # Breakout indicators
        breakout_strength = self.breakout_strength[-1]
        cascade_signal = self.cascade_signal[-1]
        acceleration_signal = self.acceleration_signal[-1]
        compression_signal = self.compression_signal[-1]
        breakout_momentum = self.breakout_momentum[-1]
        
        # Volume surge
        volume_surge = self.data.Volume[-1] / self.volume_ma[-1]
        
        # Price action
        price_change = (self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]
        
        # Calculate phase confidence
        confidence = 0.0
        
        # Compression phase
        if (compression_signal > 0.7 and breakout_strength < 0.3 and 
            volume_surge < 1.2 and abs(price_change) < 0.01):
            confidence = 0.9
            return BreakoutPhase.COMPRESSION, confidence
        
        # Initial breakout phase
        elif (breakout_strength > 0.5 and cascade_signal > 0.3 and 
              volume_surge > 1.5 and abs(price_change) > 0.01):
            confidence = 0.9
            return BreakoutPhase.INITIAL_BREAKOUT, confidence
        
        # Cascade phase
        elif (cascade_signal > self.cascade_threshold and 
              breakout_strength > 0.6 and volume_surge > 2.0 and
              abs(price_change) > 0.02):
            confidence = 0.95
            return BreakoutPhase.CASCADE, confidence
        
        # Acceleration phase
        elif (acceleration_signal > self.acceleration_threshold and 
              breakout_momentum > 0.2 and volume_surge > 2.5 and
              abs(price_change) > 0.03):
            confidence = 0.9
            return BreakoutPhase.ACCELERATION, confidence
        
        # Exhaustion phase
        elif (breakout_strength > 0.7 and cascade_signal > 0.6 and 
              acceleration_signal < 0.3 and volume_surge > 2.0 and
              abs(price_change) < 0.01):
            confidence = 0.8
            return BreakoutPhase.EXHAUSTION, confidence
        
        return BreakoutPhase.COMPRESSION, 0.5
    
    def should_enter_long_breakout(self) -> bool:
        """Determine if we should enter a long position for breakout cascade"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_breakout_phase()
        
        # Only enter during initial breakout, cascade, or acceleration phases
        if current_phase not in [BreakoutPhase.INITIAL_BREAKOUT, BreakoutPhase.CASCADE, BreakoutPhase.ACCELERATION]:
            return False
        
        # Technical conditions
        rsi_strong = self.rsi[-1] > 45
        rsi_not_overbought = self.rsi[-1] < 75
        
        # Momentum conditions
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        macd_rising = self.macd_histogram[-1] > self.macd_histogram[-2]
        
        # Volume conditions
        volume_breakout = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_breakout_threshold
        
        # Price conditions
        price_above_bb_middle = self.data.Close[-1] > self.bb_middle[-1]
        price_breakout = self.data.Close[-1] > self.bb_upper[-1]
        
        # Breakout conditions
        breakout_strength_high = self.breakout_strength[-1] > 0.5
        cascade_signal_high = self.cascade_signal[-1] > 0.3
        acceleration_signal_high = self.acceleration_signal[-1] > 0.4
        breakout_momentum_positive = self.breakout_momentum[-1] > 0.1
        
        # Stochastic confirmation
        stoch_bullish = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-1] > 50
        
        # Williams %R confirmation
        williams_bullish = self.williams_r[-1] > -50
        
        return (rsi_strong and rsi_not_overbought and
                macd_positive and macd_rising and
                volume_breakout and
                price_above_bb_middle and price_breakout and
                breakout_strength_high and cascade_signal_high and
                acceleration_signal_high and breakout_momentum_positive and
                stoch_bullish and williams_bullish and
                phase_confidence > 0.6)
    
    def should_enter_short_breakout(self) -> bool:
        """Determine if we should enter a short position for breakout reversal"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_breakout_phase()
        
        # Only enter during exhaustion phase
        if current_phase != BreakoutPhase.EXHAUSTION:
            return False
        
        # Technical conditions
        rsi_overbought = self.rsi[-1] > 70
        rsi_divergence = self.rsi[-1] < self.rsi[-2] and self.data.Close[-1] > self.data.Close[-2]
        
        # Momentum conditions
        macd_negative = self.macd[-1] < self.macd_signal_line[-1]
        macd_falling = self.macd_histogram[-1] < self.macd_histogram[-2]
        
        # Volume conditions
        volume_breakout = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_breakout_threshold
        
        # Price conditions
        price_below_bb_middle = self.data.Close[-1] < self.bb_middle[-1]
        price_breakout = self.data.Close[-1] < self.bb_lower[-1]
        
        # Breakout conditions
        breakout_strength_high = self.breakout_strength[-1] > 0.6
        cascade_signal_high = self.cascade_signal[-1] > 0.5
        acceleration_signal_low = self.acceleration_signal[-1] < 0.3
        breakout_momentum_negative = self.breakout_momentum[-1] < -0.1
        
        # Stochastic confirmation
        stoch_bearish = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-1] < 50
        
        # Williams %R confirmation
        williams_bearish = self.williams_r[-1] < -50
        
        return (rsi_overbought and rsi_divergence and
                macd_negative and macd_falling and
                volume_breakout and
                price_below_bb_middle and price_breakout and
                breakout_strength_high and cascade_signal_high and
                acceleration_signal_low and breakout_momentum_negative and
                stoch_bearish and williams_bearish and
                phase_confidence > 0.6)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on breakout strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust based on breakout strength
        breakout_strength = self.breakout_strength[-1]
        strength_multiplier = 1.0 + (breakout_strength * 0.6)  # Up to 60% increase
        
        # Adjust based on cascade signal
        cascade_signal = self.cascade_signal[-1]
        cascade_multiplier = 1.0 + (cascade_signal * 0.4)  # Up to 40% increase
        
        # Adjust based on acceleration
        acceleration_signal = self.acceleration_signal[-1]
        acceleration_multiplier = 1.0 + (acceleration_signal * 0.3)  # Up to 30% increase
        
        # Adjust based on phase
        current_phase, phase_confidence = self.detect_breakout_phase()
        phase_multiplier = 1.0
        if current_phase == BreakoutPhase.CASCADE:
            phase_multiplier = 1.5
        elif current_phase == BreakoutPhase.ACCELERATION:
            phase_multiplier = 1.3
        elif current_phase == BreakoutPhase.INITIAL_BREAKOUT:
            phase_multiplier = 1.1
        
        # Final position size
        adjusted_position_size = base_position_size * strength_multiplier * cascade_multiplier * acceleration_multiplier * phase_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss based on breakout volatility"""
        atr_value = self.atr[-1]
        
        # Adjust ATR multiplier based on breakout strength
        breakout_strength = self.breakout_strength[-1]
        atr_multiplier = self.atr_multiplier * (1.0 + breakout_strength * 0.3)
        
        stop_distance = atr_value * atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with high risk-reward for breakout moves"""
        risk = abs(entry_price - stop_loss)
        
        # Higher risk-reward ratio for breakout moves
        risk_reward_ratio = 3.0
        
        # Adjust based on cascade signal
        cascade_signal = self.cascade_signal[-1]
        if cascade_signal > 0.8:  # Very strong cascade
            risk_reward_ratio = 3.5
        elif cascade_signal > 0.6:  # Strong cascade
            risk_reward_ratio = 3.0
        else:  # Moderate breakout
            risk_reward_ratio = 2.5
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update breakout phase
        current_phase, phase_confidence = self.detect_breakout_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != current_phase.value:
            logger.info(f"Breakout phase: {current_phase.value} (confidence: {phase_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for breakout entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry for breakout cascade
        if self.should_enter_long_breakout():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_BREAKOUT", current_price, stop_loss, take_profit, position_size)
        
        # Short entry for breakout reversal
        elif self.should_enter_short_breakout():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_BREAKOUT", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with breakout-specific logic"""
        for position in self.positions:
            current_phase, phase_confidence = self.detect_breakout_phase()
            
            # Close positions during exhaustion phase
            if current_phase == BreakoutPhase.EXHAUSTION and position.pl > 0.02:
                self.position.close()
                logger.info("Closing position due to breakout exhaustion")
            
            # Close positions during compression phase
            elif current_phase == BreakoutPhase.COMPRESSION and position.pl < -0.01:
                self.position.close()
                logger.info("Closing position due to breakout compression")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        current_phase, phase_confidence = self.detect_breakout_phase()
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': current_phase.value,
            'phase_confidence': phase_confidence,
            'breakout_strength': self.breakout_strength[-1],
            'cascade_signal': self.cascade_signal[-1],
            'acceleration_signal': self.acceleration_signal[-1],
            'breakout_momentum': self.breakout_momentum[-1],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Phase={current_phase.value}, Breakout={self.breakout_strength[-1]:.3f}")
    
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
            # Reduce risk per trade temporarily
            self.risk_per_trade *= 0.8
