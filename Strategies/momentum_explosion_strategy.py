#!/usr/bin/env python3
"""
Momentum Explosion Strategy

A high-frequency strategy designed to capture extreme momentum moves during bull runs:
- Detects momentum explosions using multiple timeframes
- Uses volume profile analysis to identify institutional flows
- Implements dynamic position sizing based on momentum strength
- Focuses on capturing the most explosive moves in bull markets
- Uses advanced technical indicators for momentum confirmation

This strategy is optimized for capturing the most profitable moves during bull run conditions.
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

class MomentumPhase(Enum):
    """Momentum phases for strategic positioning"""
    ACCUMULATION = "accumulation"      # Early momentum building
    EXPLOSION = "explosion"            # Main momentum explosion
    ACCELERATION = "acceleration"      # Momentum acceleration
    EXHAUSTION = "exhaustion"          # Momentum exhaustion
    REVERSAL = "reversal"              # Momentum reversal

class MomentumExplosionStrategy(Strategy):
    """
    Momentum Explosion Strategy
    
    Captures extreme momentum moves during bull runs:
    - Multi-timeframe momentum analysis
    - Volume profile analysis for institutional flows
    - Dynamic position sizing based on momentum strength
    - Advanced risk management for volatile conditions
    """
    
    # Core parameters
    risk_per_trade = 0.015             # 1.5% risk per trade
    max_positions = 4                  # Maximum concurrent positions
    max_drawdown = 0.12                # Maximum drawdown limit
    consecutive_loss_limit = 4         # Maximum consecutive losses
    
    # Momentum detection parameters
    momentum_lookback = 30             # Periods to analyze for momentum
    explosion_threshold = 0.03         # Price explosion threshold
    volume_explosion_threshold = 2.5   # Volume explosion threshold
    momentum_acceleration_threshold = 0.02 # Momentum acceleration threshold
    
    # Multi-timeframe analysis
    timeframes = [5, 15, 30, 60]       # Different timeframes for analysis
    momentum_periods = [8, 14, 21, 34] # RSI periods for different timeframes
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 35
    rsi_overbought = 65
    atr_period = 14
    atr_multiplier = 2.0
    
    # MACD parameters
    macd_fast = 8
    macd_slow = 21
    macd_signal = 5
    
    # Volume analysis
    volume_period = 20
    volume_profile_period = 50
    
    # Momentum indicators
    momentum_period = 10
    rate_of_change_period = 5
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.momentum_phases = []
        self.explosion_signals = []
        
        # Multi-timeframe RSI
        self.rsi_5 = self.I(talib.RSI, self.data.Close, timeperiod=5)
        self.rsi_14 = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.rsi_21 = self.I(talib.RSI, self.data.Close, timeperiod=21)
        self.rsi_34 = self.I(talib.RSI, self.data.Close, timeperiod=34)
        
        # MACD for momentum
        self.macd, self.macd_signal_line, self.macd_histogram = self.I(
            talib.MACD, self.data.Close, 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        # ATR for volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        self.volume_profile = self.I(self._calculate_volume_profile)
        
        # Momentum indicators
        self.momentum = self.I(talib.MOM, self.data.Close, timeperiod=self.momentum_period)
        self.roc = self.I(talib.ROC, self.data.Close, timeperiod=self.rate_of_change_period)
        
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
        
        # Momentum explosion indicator
        self.momentum_explosion = self.I(self._calculate_momentum_explosion)
        
        # Volume explosion indicator
        self.volume_explosion = self.I(self._calculate_volume_explosion)
    
    def _calculate_volume_profile(self, data):
        """Calculate volume profile for institutional flow analysis"""
        if len(data) < self.volume_profile_period:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate volume-weighted average price
        vwap = (data.Close * data.Volume).rolling(self.volume_profile_period).sum() / data.Volume.rolling(self.volume_profile_period).sum()
        
        # Calculate volume profile strength
        volume_profile_strength = abs(data.Close - vwap) / vwap
        
        return volume_profile_strength.rolling(window=5).mean()
    
    def _calculate_momentum_explosion(self, data):
        """Calculate momentum explosion indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Price momentum
        price_momentum = data.Close.pct_change(5)
        
        # Volume momentum
        volume_momentum = data.Volume.pct_change(5)
        
        # RSI momentum
        rsi_momentum = self.rsi_14.diff(3)
        
        # MACD momentum
        macd_momentum = self.macd_histogram.diff(2)
        
        # Combine momentum factors
        momentum_explosion = (
            abs(price_momentum) * 0.4 +
            abs(volume_momentum) * 0.3 +
            abs(rsi_momentum) * 0.2 +
            abs(macd_momentum) * 0.1
        )
        
        return momentum_explosion.rolling(window=3).mean()
    
    def _calculate_volume_explosion(self, data):
        """Calculate volume explosion indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Volume ratio
        volume_ratio = data.Volume / self.volume_ma
        
        # Volume acceleration
        volume_acceleration = data.Volume.pct_change().rolling(3).mean()
        
        # Volume profile strength
        volume_profile_strength = self.volume_profile
        
        # Combine volume factors
        volume_explosion = (
            volume_ratio * 0.5 +
            abs(volume_acceleration) * 0.3 +
            volume_profile_strength * 0.2
        )
        
        return volume_explosion.rolling(window=3).mean()
    
    def detect_momentum_phase(self) -> Tuple[MomentumPhase, float]:
        """Detect the current momentum phase"""
        if len(self.data) < max(self.momentum_periods):
            return MomentumPhase.ACCUMULATION, 0.5
        
        # Multi-timeframe RSI analysis
        rsi_5_val = self.rsi_5[-1]
        rsi_14_val = self.rsi_14[-1]
        rsi_21_val = self.rsi_21[-1]
        rsi_34_val = self.rsi_34[-1]
        
        # Momentum indicators
        momentum_val = self.momentum[-1]
        roc_val = self.roc[-1]
        macd_hist_val = self.macd_histogram[-1]
        
        # Volume analysis
        volume_ratio = self.data.Volume[-1] / self.volume_ma[-1]
        volume_explosion_val = self.volume_explosion[-1]
        
        # Price action
        price_change = (self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]
        
        # Calculate phase confidence
        confidence = 0.0
        
        # Explosion phase detection
        if (rsi_5_val > 70 and rsi_14_val > 60 and rsi_21_val > 50 and 
            momentum_val > 0 and roc_val > 0.02 and volume_ratio > 2.0):
            confidence = 0.9
            return MomentumPhase.EXPLOSION, confidence
        
        # Acceleration phase detection
        elif (rsi_5_val > 60 and rsi_14_val > 50 and rsi_21_val > 45 and 
              momentum_val > 0 and macd_hist_val > 0 and volume_ratio > 1.5):
            confidence = 0.8
            return MomentumPhase.ACCELERATION, confidence
        
        # Accumulation phase detection
        elif (rsi_5_val > 40 and rsi_14_val > 35 and rsi_21_val > 30 and 
              momentum_val > -0.01 and volume_ratio > 1.0):
            confidence = 0.7
            return MomentumPhase.ACCUMULATION, confidence
        
        # Exhaustion phase detection
        elif (rsi_5_val > 80 and rsi_14_val > 75 and rsi_21_val > 70 and 
              momentum_val < 0 and volume_ratio > 2.5):
            confidence = 0.8
            return MomentumPhase.EXHAUSTION, confidence
        
        # Reversal phase detection
        elif (rsi_5_val < 30 and rsi_14_val < 40 and rsi_21_val < 45 and 
              momentum_val < -0.01 and volume_ratio > 1.5):
            confidence = 0.7
            return MomentumPhase.REVERSAL, confidence
        
        return MomentumPhase.ACCUMULATION, 0.5
    
    def should_enter_long_momentum(self) -> bool:
        """Determine if we should enter a long position for momentum explosion"""
        if len(self.data) < max(self.momentum_periods):
            return False
        
        current_phase, phase_confidence = self.detect_momentum_phase()
        
        # Only enter during accumulation, acceleration, or explosion phases
        if current_phase not in [MomentumPhase.ACCUMULATION, MomentumPhase.ACCELERATION, MomentumPhase.EXPLOSION]:
            return False
        
        # Technical conditions
        rsi_5_strong = self.rsi_5[-1] > 45
        rsi_14_strong = self.rsi_14[-1] > 40
        rsi_21_strong = self.rsi_21[-1] > 35
        
        # Momentum conditions
        momentum_positive = self.momentum[-1] > 0
        roc_positive = self.roc[-1] > 0.01
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        macd_rising = self.macd_histogram[-1] > self.macd_histogram[-2]
        
        # Volume conditions
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * 1.5
        volume_explosion_high = self.volume_explosion[-1] > 1.0
        
        # Price action
        price_above_bb_middle = self.data.Close[-1] > self.bb_middle[-1]
        price_momentum_strong = self.momentum_explosion[-1] > 0.02
        
        # Stochastic confirmation
        stoch_bullish = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-1] > 50
        
        # Williams %R confirmation
        williams_bullish = self.williams_r[-1] > -50
        
        return (rsi_5_strong and rsi_14_strong and rsi_21_strong and
                momentum_positive and roc_positive and macd_positive and macd_rising and
                volume_strong and volume_explosion_high and
                price_above_bb_middle and price_momentum_strong and
                stoch_bullish and williams_bullish and
                phase_confidence > 0.6)
    
    def should_enter_short_momentum(self) -> bool:
        """Determine if we should enter a short position for momentum reversal"""
        if len(self.data) < max(self.momentum_periods):
            return False
        
        current_phase, phase_confidence = self.detect_momentum_phase()
        
        # Only enter during exhaustion or reversal phases
        if current_phase not in [MomentumPhase.EXHAUSTION, MomentumPhase.REVERSAL]:
            return False
        
        # Technical conditions
        rsi_5_overbought = self.rsi_5[-1] > 75
        rsi_14_overbought = self.rsi_14[-1] > 70
        rsi_21_overbought = self.rsi_21[-1] > 65
        
        # Momentum conditions
        momentum_negative = self.momentum[-1] < 0
        roc_negative = self.roc[-1] < -0.01
        macd_negative = self.macd[-1] < self.macd_signal_line[-1]
        macd_falling = self.macd_histogram[-1] < self.macd_histogram[-2]
        
        # Volume conditions
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * 1.5
        volume_explosion_high = self.volume_explosion[-1] > 1.0
        
        # Price action
        price_below_bb_middle = self.data.Close[-1] < self.bb_middle[-1]
        price_momentum_weak = self.momentum_explosion[-1] < -0.02
        
        # Stochastic confirmation
        stoch_bearish = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-1] < 50
        
        # Williams %R confirmation
        williams_bearish = self.williams_r[-1] < -50
        
        return (rsi_5_overbought and rsi_14_overbought and rsi_21_overbought and
                momentum_negative and roc_negative and macd_negative and macd_falling and
                volume_strong and volume_explosion_high and
                price_below_bb_middle and price_momentum_weak and
                stoch_bearish and williams_bearish and
                phase_confidence > 0.6)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on momentum strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust based on momentum strength
        momentum_strength = self.momentum_explosion[-1]
        momentum_multiplier = 1.0 + (momentum_strength * 2.0)  # Up to 2x increase
        
        # Adjust based on volume explosion
        volume_strength = self.volume_explosion[-1]
        volume_multiplier = 1.0 + (volume_strength * 0.5)  # Up to 50% increase
        
        # Adjust based on phase
        current_phase, phase_confidence = self.detect_momentum_phase()
        phase_multiplier = 1.0
        if current_phase == MomentumPhase.EXPLOSION:
            phase_multiplier = 1.5
        elif current_phase == MomentumPhase.ACCELERATION:
            phase_multiplier = 1.2
        elif current_phase == MomentumPhase.ACCUMULATION:
            phase_multiplier = 0.8
        
        # Final position size
        adjusted_position_size = base_position_size * momentum_multiplier * volume_multiplier * phase_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss based on momentum volatility"""
        atr_value = self.atr[-1]
        
        # Adjust ATR multiplier based on momentum
        momentum_strength = self.momentum_explosion[-1]
        atr_multiplier = self.atr_multiplier * (1.0 + momentum_strength)
        
        stop_distance = atr_value * atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with high risk-reward for momentum moves"""
        risk = abs(entry_price - stop_loss)
        
        # Higher risk-reward ratio for momentum moves
        risk_reward_ratio = 2.5
        
        # Adjust based on momentum strength
        momentum_strength = self.momentum_explosion[-1]
        if momentum_strength > 0.05:  # Very strong momentum
            risk_reward_ratio = 3.0
        elif momentum_strength > 0.03:  # Strong momentum
            risk_reward_ratio = 2.5
        else:  # Moderate momentum
            risk_reward_ratio = 2.0
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update momentum phase
        current_phase, phase_confidence = self.detect_momentum_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != current_phase.value:
            logger.info(f"Momentum phase: {current_phase.value} (confidence: {phase_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for momentum entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry for momentum explosion
        if self.should_enter_long_momentum():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_MOMENTUM", current_price, stop_loss, take_profit, position_size)
        
        # Short entry for momentum reversal
        elif self.should_enter_short_momentum():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_MOMENTUM", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with momentum-specific logic"""
        for position in self.positions:
            current_phase, phase_confidence = self.detect_momentum_phase()
            
            # Close positions during exhaustion phase
            if current_phase == MomentumPhase.EXHAUSTION and position.pl > 0.02:
                self.position.close()
                logger.info("Closing position due to momentum exhaustion")
            
            # Close positions during reversal phase
            elif current_phase == MomentumPhase.REVERSAL and position.pl < -0.01:
                self.position.close()
                logger.info("Closing position due to momentum reversal")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        current_phase, phase_confidence = self.detect_momentum_phase()
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': current_phase.value,
            'phase_confidence': phase_confidence,
            'momentum_strength': self.momentum_explosion[-1],
            'volume_strength': self.volume_explosion[-1],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Phase={current_phase.value}, Momentum={self.momentum_explosion[-1]:.3f}")
    
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
