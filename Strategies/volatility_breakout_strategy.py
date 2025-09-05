#!/usr/bin/env python3
"""
Volatility Breakout Strategy

A high-frequency strategy designed to capture explosive volatility breakouts during bull runs:
- Detects volatility compression and expansion patterns
- Identifies breakout opportunities with high probability
- Uses multiple volatility indicators for confirmation
- Implements aggressive position sizing for breakout events
- Focuses on capturing the most volatile moves in bull markets

This strategy targets the most explosive price movements during bull run conditions.
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

class VolatilityPhase(Enum):
    """Volatility phases for strategic positioning"""
    COMPRESSION = "compression"        # Low volatility, building pressure
    EXPANSION = "expansion"            # Volatility increasing
    BREAKOUT = "breakout"              # Volatility breakout
    EXPLOSION = "explosion"            # Extreme volatility
    EXHAUSTION = "exhaustion"          # Volatility exhaustion

class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy
    
    Captures explosive volatility breakouts during bull runs:
    - Detects volatility compression and expansion patterns
    - Identifies breakout opportunities with high probability
    - Uses multiple volatility indicators for confirmation
    - Implements aggressive position sizing for breakout events
    """
    
    # Core parameters
    risk_per_trade = 0.018             # 1.8% risk per trade (aggressive for breakouts)
    max_positions = 4                  # Maximum concurrent positions
    max_drawdown = 0.14                # Maximum drawdown limit
    consecutive_loss_limit = 3         # Maximum consecutive losses
    
    # Volatility detection parameters
    volatility_lookback = 30           # Periods to analyze for volatility patterns
    compression_threshold = 0.5        # Volatility compression threshold
    expansion_threshold = 1.5          # Volatility expansion threshold
    breakout_threshold = 2.0           # Volatility breakout threshold
    
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
    
    # Volatility indicators
    volatility_period = 20
    bollinger_period = 20
    bollinger_std = 2.0
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.volatility_phases = []
        self.breakout_signals = []
        
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
            talib.BBANDS, self.data.Close, timeperiod=self.bollinger_period, 
            nbdevup=self.bollinger_std, nbdevdn=self.bollinger_std
        )
        
        # Stochastic for momentum
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, self.data.High, self.data.Low, self.data.Close,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R for momentum
        self.williams_r = self.I(talib.WILLR, self.data.High, self.data.Low, 
                                self.data.Close, timeperiod=14)
        
        # Volatility indicators
        self.volatility = self.I(self._calculate_volatility)
        self.volatility_ratio = self.I(self._calculate_volatility_ratio)
        self.volatility_compression = self.I(self._calculate_volatility_compression)
        self.volatility_expansion = self.I(self._calculate_volatility_expansion)
        self.breakout_strength = self.I(self._calculate_breakout_strength)
    
    def _calculate_volatility(self, data):
        """Calculate historical volatility"""
        if len(data) < self.volatility_period:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate returns
        returns = data.Close.pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.volatility_period).std() * np.sqrt(252)
        
        return volatility.rolling(window=5).mean()
    
    def _calculate_volatility_ratio(self, data):
        """Calculate volatility ratio (current vs historical)"""
        if len(data) < self.volatility_period * 2:
            return pd.Series([0] * len(data), index=data.index)
        
        # Current volatility
        current_volatility = self.volatility
        
        # Historical volatility
        historical_volatility = current_volatility.rolling(window=self.volatility_period).mean()
        
        # Volatility ratio
        volatility_ratio = current_volatility / historical_volatility
        
        return volatility_ratio.rolling(window=3).mean()
    
    def _calculate_volatility_compression(self, data):
        """Calculate volatility compression indicator"""
        if len(data) < self.volatility_period:
            return pd.Series([0] * len(data), index=data.index)
        
        # Bollinger Band width
        bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        
        # ATR relative to price
        atr_relative = self.atr / data.Close
        
        # Volatility compression (low volatility)
        compression = (bb_width < bb_width.rolling(20).mean() * 0.8) & (atr_relative < atr_relative.rolling(20).mean() * 0.8)
        
        return compression.astype(int).rolling(window=5).mean()
    
    def _calculate_volatility_expansion(self, data):
        """Calculate volatility expansion indicator"""
        if len(data) < self.volatility_period:
            return pd.Series([0] * len(data), index=data.index)
        
        # Bollinger Band width
        bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        
        # ATR relative to price
        atr_relative = self.atr / data.Close
        
        # Volatility expansion (high volatility)
        expansion = (bb_width > bb_width.rolling(20).mean() * 1.2) | (atr_relative > atr_relative.rolling(20).mean() * 1.2)
        
        return expansion.astype(int).rolling(window=3).mean()
    
    def _calculate_breakout_strength(self, data):
        """Calculate breakout strength indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Price momentum
        price_momentum = data.Close.pct_change(3)
        
        # Volume surge
        volume_surge = data.Volume / self.volume_ma
        
        # Volatility expansion
        volatility_expansion = self.volatility_expansion
        
        # RSI momentum
        rsi_momentum = self.rsi.diff(2)
        
        # Combine breakout factors
        breakout_strength = (
            abs(price_momentum) * 0.3 +
            volume_surge * 0.3 +
            volatility_expansion * 0.2 +
            abs(rsi_momentum) * 0.2
        )
        
        return breakout_strength.rolling(window=3).mean()
    
    def detect_volatility_phase(self) -> Tuple[VolatilityPhase, float]:
        """Detect the current volatility phase"""
        if len(self.data) < self.volatility_lookback:
            return VolatilityPhase.COMPRESSION, 0.5
        
        # Volatility indicators
        volatility_ratio = self.volatility_ratio[-1]
        compression = self.volatility_compression[-1]
        expansion = self.volatility_expansion[-1]
        breakout_strength = self.breakout_strength[-1]
        
        # Volume surge
        volume_surge = self.data.Volume[-1] / self.volume_ma[-1]
        
        # Price action
        price_change = (self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]
        
        # Calculate phase confidence
        confidence = 0.0
        
        # Compression phase
        if (compression > 0.7 and volatility_ratio < 0.8 and 
            volume_surge < 1.2 and abs(price_change) < 0.02):
            confidence = 0.9
            return VolatilityPhase.COMPRESSION, confidence
        
        # Expansion phase
        elif (expansion > 0.6 and volatility_ratio > 1.2 and 
              volume_surge > 1.5 and abs(price_change) > 0.01):
            confidence = 0.8
            return VolatilityPhase.EXPANSION, confidence
        
        # Breakout phase
        elif (breakout_strength > 0.6 and volatility_ratio > 1.5 and 
              volume_surge > 2.0 and abs(price_change) > 0.02):
            confidence = 0.9
            return VolatilityPhase.BREAKOUT, confidence
        
        # Explosion phase
        elif (breakout_strength > 0.8 and volatility_ratio > 2.0 and 
              volume_surge > 3.0 and abs(price_change) > 0.03):
            confidence = 0.95
            return VolatilityPhase.EXPLOSION, confidence
        
        # Exhaustion phase
        elif (breakout_strength > 0.7 and volatility_ratio > 1.8 and 
              volume_surge > 2.5 and abs(price_change) < 0.01):
            confidence = 0.8
            return VolatilityPhase.EXHAUSTION, confidence
        
        return VolatilityPhase.COMPRESSION, 0.5
    
    def should_enter_long_breakout(self) -> bool:
        """Determine if we should enter a long position for volatility breakout"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_volatility_phase()
        
        # Only enter during expansion, breakout, or explosion phases
        if current_phase not in [VolatilityPhase.EXPANSION, VolatilityPhase.BREAKOUT, VolatilityPhase.EXPLOSION]:
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
        
        # Volatility conditions
        volatility_expansion_high = self.volatility_expansion[-1] > 0.6
        breakout_strength_high = self.breakout_strength[-1] > 0.5
        volatility_ratio_high = self.volatility_ratio[-1] > 1.2
        
        # Stochastic confirmation
        stoch_bullish = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-1] > 50
        
        # Williams %R confirmation
        williams_bullish = self.williams_r[-1] > -50
        
        return (rsi_strong and rsi_not_overbought and
                macd_positive and macd_rising and
                volume_breakout and
                price_above_bb_middle and price_breakout and
                volatility_expansion_high and breakout_strength_high and volatility_ratio_high and
                stoch_bullish and williams_bullish and
                phase_confidence > 0.6)
    
    def should_enter_short_breakout(self) -> bool:
        """Determine if we should enter a short position for volatility breakout"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_volatility_phase()
        
        # Only enter during expansion, breakout, or explosion phases
        if current_phase not in [VolatilityPhase.EXPANSION, VolatilityPhase.BREAKOUT, VolatilityPhase.EXPLOSION]:
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
        
        # Volatility conditions
        volatility_expansion_high = self.volatility_expansion[-1] > 0.6
        breakout_strength_high = self.breakout_strength[-1] > 0.5
        volatility_ratio_high = self.volatility_ratio[-1] > 1.2
        
        # Stochastic confirmation
        stoch_bearish = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-1] < 50
        
        # Williams %R confirmation
        williams_bearish = self.williams_r[-1] < -50
        
        return (rsi_overbought and rsi_divergence and
                macd_negative and macd_falling and
                volume_breakout and
                price_below_bb_middle and price_breakout and
                volatility_expansion_high and breakout_strength_high and volatility_ratio_high and
                stoch_bearish and williams_bearish and
                phase_confidence > 0.6)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on volatility breakout strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust based on breakout strength
        breakout_strength = self.breakout_strength[-1]
        breakout_multiplier = 1.0 + (breakout_strength * 1.0)  # Up to 100% increase
        
        # Adjust based on volatility ratio
        volatility_ratio = self.volatility_ratio[-1]
        volatility_multiplier = 1.0 + (volatility_ratio - 1.0) * 0.5  # Up to 50% increase
        
        # Adjust based on phase
        current_phase, phase_confidence = self.detect_volatility_phase()
        phase_multiplier = 1.0
        if current_phase == VolatilityPhase.EXPLOSION:
            phase_multiplier = 1.5
        elif current_phase == VolatilityPhase.BREAKOUT:
            phase_multiplier = 1.2
        elif current_phase == VolatilityPhase.EXPANSION:
            phase_multiplier = 0.8
        
        # Final position size
        adjusted_position_size = base_position_size * breakout_multiplier * volatility_multiplier * phase_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss based on volatility breakout"""
        atr_value = self.atr[-1]
        
        # Adjust ATR multiplier based on volatility
        volatility_ratio = self.volatility_ratio[-1]
        atr_multiplier = self.atr_multiplier * (1.0 + volatility_ratio * 0.3)
        
        stop_distance = atr_value * atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with high risk-reward for volatility breakouts"""
        risk = abs(entry_price - stop_loss)
        
        # Higher risk-reward ratio for volatility breakouts
        risk_reward_ratio = 2.5
        
        # Adjust based on breakout strength
        breakout_strength = self.breakout_strength[-1]
        if breakout_strength > 0.8:  # Very strong breakout
            risk_reward_ratio = 3.0
        elif breakout_strength > 0.6:  # Strong breakout
            risk_reward_ratio = 2.5
        else:  # Moderate breakout
            risk_reward_ratio = 2.0
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update volatility phase
        current_phase, phase_confidence = self.detect_volatility_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != current_phase.value:
            logger.info(f"Volatility phase: {current_phase.value} (confidence: {phase_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for volatility breakout entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry for volatility breakout
        if self.should_enter_long_breakout():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_BREAKOUT", current_price, stop_loss, take_profit, position_size)
        
        # Short entry for volatility breakout
        elif self.should_enter_short_breakout():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_BREAKOUT", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with volatility-specific logic"""
        for position in self.positions:
            current_phase, phase_confidence = self.detect_volatility_phase()
            
            # Close positions during exhaustion phase
            if current_phase == VolatilityPhase.EXHAUSTION and position.pl > 0.02:
                self.position.close()
                logger.info("Closing position due to volatility exhaustion")
            
            # Close positions during compression phase
            elif current_phase == VolatilityPhase.COMPRESSION and position.pl < -0.01:
                self.position.close()
                logger.info("Closing position due to volatility compression")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        current_phase, phase_confidence = self.detect_volatility_phase()
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': current_phase.value,
            'phase_confidence': phase_confidence,
            'breakout_strength': self.breakout_strength[-1],
            'volatility_ratio': self.volatility_ratio[-1],
            'volatility_expansion': self.volatility_expansion[-1],
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
