#!/usr/bin/env python3
"""
FOMO Capture Strategy

A strategy designed to capitalize on FOMO (Fear of Missing Out) buying waves during bull runs:
- Detects retail FOMO buying patterns through volume and price analysis
- Identifies social sentiment indicators and news catalysts
- Uses order flow analysis to detect retail vs institutional activity
- Implements aggressive position sizing for FOMO events
- Focuses on capturing the most emotional and profitable moves

This strategy targets the psychological aspects of bull market behavior.
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

class FOMOPhase(Enum):
    """FOMO phases for strategic positioning"""
    EARLY_FOMO = "early_fomo"          # Early retail buying
    MASS_FOMO = "mass_fomo"            # Mass retail participation
    PANIC_FOMO = "panic_fomo"          # Panic buying at peaks
    EXHAUSTION = "exhaustion"           # FOMO exhaustion
    REVERSAL = "reversal"              # FOMO reversal

class FOMOCaptureStrategy(Strategy):
    """
    FOMO Capture Strategy
    
    Captures FOMO buying waves during bull runs:
    - Detects retail FOMO patterns through volume/price analysis
    - Uses social sentiment and news catalysts
    - Implements aggressive position sizing for FOMO events
    - Advanced risk management for emotional market conditions
    """
    
    # Core parameters
    risk_per_trade = 0.02              # 2% risk per trade (aggressive for FOMO)
    max_positions = 5                  # Maximum concurrent positions
    max_drawdown = 0.15                # Maximum drawdown limit
    consecutive_loss_limit = 3         # Maximum consecutive losses
    
    # FOMO detection parameters
    fomo_lookback = 20                 # Periods to analyze for FOMO patterns
    volume_surge_threshold = 2.0       # Volume surge threshold for FOMO
    price_acceleration_threshold = 0.02 # Price acceleration threshold
    retail_flow_threshold = 0.7        # Retail flow threshold
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    atr_period = 14
    atr_multiplier = 2.5
    
    # MACD parameters
    macd_fast = 8
    macd_slow = 21
    macd_signal = 5
    
    # Volume analysis
    volume_period = 20
    volume_surge_period = 5
    
    # FOMO indicators
    fomo_strength_period = 10
    social_sentiment_period = 7
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.fomo_phases = []
        self.retail_flow_indicators = []
        
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
        self.volume_surge_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_surge_period)
        
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
        
        # FOMO indicators
        self.fomo_strength = self.I(self._calculate_fomo_strength)
        self.retail_flow = self.I(self._calculate_retail_flow)
        self.social_sentiment = self.I(self._calculate_social_sentiment)
        self.price_acceleration = self.I(self._calculate_price_acceleration)
        self.volume_acceleration = self.I(self._calculate_volume_acceleration)
    
    def _calculate_fomo_strength(self, data):
        """Calculate FOMO strength indicator"""
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
        
        # Combine FOMO factors
        fomo_strength = (
            abs(price_momentum) * 0.3 +
            volume_surge * 0.3 +
            abs(rsi_momentum) * 0.2 +
            abs(macd_momentum) * 0.2
        )
        
        return fomo_strength.rolling(window=self.fomo_strength_period).mean()
    
    def _calculate_retail_flow(self, data):
        """Calculate retail flow indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Small volume trades (retail characteristic)
        volume_ratio = data.Volume / self.volume_ma
        small_volume_trades = (volume_ratio > 1.0) & (volume_ratio < 3.0)
        
        # Price volatility (retail characteristic)
        price_volatility = data.Close.pct_change().rolling(3).std()
        high_volatility = price_volatility > price_volatility.rolling(20).mean() * 1.5
        
        # RSI extremes (retail characteristic)
        rsi_extremes = (self.rsi < 25) | (self.rsi > 75)
        
        # Combine retail flow factors
        retail_flow = (
            small_volume_trades.astype(int) * 0.4 +
            high_volatility.astype(int) * 0.3 +
            rsi_extremes.astype(int) * 0.3
        )
        
        return retail_flow.rolling(window=5).mean()
    
    def _calculate_social_sentiment(self, data):
        """Calculate social sentiment indicator (simulated)"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Simulate social sentiment based on price action and volume
        price_momentum = data.Close.pct_change(5)
        volume_momentum = data.Volume.pct_change(5)
        
        # Social sentiment increases with positive momentum and volume
        social_sentiment = (
            price_momentum * 0.6 +
            volume_momentum * 0.4
        )
        
        # Normalize to 0-1 range
        social_sentiment = (social_sentiment - social_sentiment.min()) / (social_sentiment.max() - social_sentiment.min())
        
        return social_sentiment.rolling(window=self.social_sentiment_period).mean()
    
    def _calculate_price_acceleration(self, data):
        """Calculate price acceleration for FOMO detection"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate second derivative of price
        price_change = data.Close.pct_change()
        acceleration = price_change.diff()
        
        return acceleration.rolling(window=3).mean()
    
    def _calculate_volume_acceleration(self, data):
        """Calculate volume acceleration for FOMO detection"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate second derivative of volume
        volume_change = data.Volume.pct_change()
        acceleration = volume_change.diff()
        
        return acceleration.rolling(window=3).mean()
    
    def detect_fomo_phase(self) -> Tuple[FOMOPhase, float]:
        """Detect the current FOMO phase"""
        if len(self.data) < self.fomo_lookback:
            return FOMOPhase.EARLY_FOMO, 0.5
        
        # FOMO indicators
        fomo_strength = self.fomo_strength[-1]
        retail_flow = self.retail_flow[-1]
        social_sentiment = self.social_sentiment[-1]
        price_acceleration = self.price_acceleration[-1]
        volume_acceleration = self.volume_acceleration[-1]
        
        # Volume surge
        volume_surge = self.data.Volume[-1] / self.volume_ma[-1]
        
        # Price action
        price_change = (self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]
        
        # Calculate phase confidence
        confidence = 0.0
        
        # Early FOMO phase
        if (fomo_strength > 0.3 and retail_flow > 0.4 and social_sentiment > 0.5 and 
            price_acceleration > 0.01 and volume_surge > 1.5):
            confidence = 0.8
            return FOMOPhase.EARLY_FOMO, confidence
        
        # Mass FOMO phase
        elif (fomo_strength > 0.5 and retail_flow > 0.6 and social_sentiment > 0.7 and 
              price_acceleration > 0.02 and volume_surge > 2.0):
            confidence = 0.9
            return FOMOPhase.MASS_FOMO, confidence
        
        # Panic FOMO phase
        elif (fomo_strength > 0.7 and retail_flow > 0.8 and social_sentiment > 0.8 and 
              price_acceleration > 0.03 and volume_surge > 3.0):
            confidence = 0.95
            return FOMOPhase.PANIC_FOMO, confidence
        
        # Exhaustion phase
        elif (fomo_strength > 0.6 and retail_flow > 0.7 and social_sentiment > 0.6 and 
              price_acceleration < 0.01 and volume_surge > 2.5):
            confidence = 0.8
            return FOMOPhase.EXHAUSTION, confidence
        
        # Reversal phase
        elif (fomo_strength < 0.3 and retail_flow < 0.4 and social_sentiment < 0.4 and 
              price_acceleration < -0.01 and volume_surge < 1.0):
            confidence = 0.7
            return FOMOPhase.REVERSAL, confidence
        
        return FOMOPhase.EARLY_FOMO, 0.5
    
    def should_enter_long_fomo(self) -> bool:
        """Determine if we should enter a long position for FOMO capture"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_fomo_phase()
        
        # Only enter during FOMO phases
        if current_phase not in [FOMOPhase.EARLY_FOMO, FOMOPhase.MASS_FOMO, FOMOPhase.PANIC_FOMO]:
            return False
        
        # Technical conditions
        rsi_strong = self.rsi[-1] > 45
        rsi_not_overbought = self.rsi[-1] < 80
        
        # Momentum conditions
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        macd_rising = self.macd_histogram[-1] > self.macd_histogram[-2]
        
        # Volume conditions
        volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_surge_threshold
        volume_acceleration_positive = self.volume_acceleration[-1] > 0
        
        # Price conditions
        price_above_bb_middle = self.data.Close[-1] > self.bb_middle[-1]
        price_acceleration_positive = self.price_acceleration[-1] > self.price_acceleration_threshold
        
        # FOMO conditions
        fomo_strength_high = self.fomo_strength[-1] > 0.4
        retail_flow_high = self.retail_flow[-1] > self.retail_flow_threshold
        social_sentiment_high = self.social_sentiment[-1] > 0.6
        
        # Stochastic confirmation
        stoch_bullish = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-1] > 50
        
        # Williams %R confirmation
        williams_bullish = self.williams_r[-1] > -50
        
        return (rsi_strong and rsi_not_overbought and
                macd_positive and macd_rising and
                volume_surge and volume_acceleration_positive and
                price_above_bb_middle and price_acceleration_positive and
                fomo_strength_high and retail_flow_high and social_sentiment_high and
                stoch_bullish and williams_bullish and
                phase_confidence > 0.6)
    
    def should_enter_short_fomo(self) -> bool:
        """Determine if we should enter a short position for FOMO reversal"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_fomo_phase()
        
        # Only enter during exhaustion or reversal phases
        if current_phase not in [FOMOPhase.EXHAUSTION, FOMOPhase.REVERSAL]:
            return False
        
        # Technical conditions
        rsi_overbought = self.rsi[-1] > 75
        rsi_divergence = self.rsi[-1] < self.rsi[-2] and self.data.Close[-1] > self.data.Close[-2]
        
        # Momentum conditions
        macd_negative = self.macd[-1] < self.macd_signal_line[-1]
        macd_falling = self.macd_histogram[-1] < self.macd_histogram[-2]
        
        # Volume conditions
        volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * 2.0
        volume_acceleration_negative = self.volume_acceleration[-1] < 0
        
        # Price conditions
        price_below_bb_middle = self.data.Close[-1] < self.bb_middle[-1]
        price_acceleration_negative = self.price_acceleration[-1] < -0.01
        
        # FOMO conditions
        fomo_strength_high = self.fomo_strength[-1] > 0.6
        retail_flow_high = self.retail_flow[-1] > 0.8
        social_sentiment_high = self.social_sentiment[-1] > 0.7
        
        # Stochastic confirmation
        stoch_bearish = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-1] < 50
        
        # Williams %R confirmation
        williams_bearish = self.williams_r[-1] < -50
        
        return (rsi_overbought and rsi_divergence and
                macd_negative and macd_falling and
                volume_surge and volume_acceleration_negative and
                price_below_bb_middle and price_acceleration_negative and
                fomo_strength_high and retail_flow_high and social_sentiment_high and
                stoch_bearish and williams_bearish and
                phase_confidence > 0.6)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on FOMO strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust based on FOMO strength
        fomo_strength = self.fomo_strength[-1]
        fomo_multiplier = 1.0 + (fomo_strength * 1.5)  # Up to 150% increase
        
        # Adjust based on retail flow
        retail_flow = self.retail_flow[-1]
        retail_multiplier = 1.0 + (retail_flow * 0.5)  # Up to 50% increase
        
        # Adjust based on social sentiment
        social_sentiment = self.social_sentiment[-1]
        sentiment_multiplier = 1.0 + (social_sentiment * 0.3)  # Up to 30% increase
        
        # Adjust based on phase
        current_phase, phase_confidence = self.detect_fomo_phase()
        phase_multiplier = 1.0
        if current_phase == FOMOPhase.PANIC_FOMO:
            phase_multiplier = 1.5
        elif current_phase == FOMOPhase.MASS_FOMO:
            phase_multiplier = 1.2
        elif current_phase == FOMOPhase.EARLY_FOMO:
            phase_multiplier = 0.8
        
        # Final position size
        adjusted_position_size = base_position_size * fomo_multiplier * retail_multiplier * sentiment_multiplier * phase_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss based on FOMO volatility"""
        atr_value = self.atr[-1]
        
        # Adjust ATR multiplier based on FOMO strength
        fomo_strength = self.fomo_strength[-1]
        atr_multiplier = self.atr_multiplier * (1.0 + fomo_strength)
        
        stop_distance = atr_value * atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with high risk-reward for FOMO moves"""
        risk = abs(entry_price - stop_loss)
        
        # Higher risk-reward ratio for FOMO moves
        risk_reward_ratio = 2.0
        
        # Adjust based on FOMO strength
        fomo_strength = self.fomo_strength[-1]
        if fomo_strength > 0.7:  # Very strong FOMO
            risk_reward_ratio = 2.5
        elif fomo_strength > 0.5:  # Strong FOMO
            risk_reward_ratio = 2.0
        else:  # Moderate FOMO
            risk_reward_ratio = 1.5
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update FOMO phase
        current_phase, phase_confidence = self.detect_fomo_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != current_phase.value:
            logger.info(f"FOMO phase: {current_phase.value} (confidence: {phase_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for FOMO entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry for FOMO capture
        if self.should_enter_long_fomo():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_FOMO", current_price, stop_loss, take_profit, position_size)
        
        # Short entry for FOMO reversal
        elif self.should_enter_short_fomo():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_FOMO", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with FOMO-specific logic"""
        for position in self.positions:
            current_phase, phase_confidence = self.detect_fomo_phase()
            
            # Close positions during exhaustion phase
            if current_phase == FOMOPhase.EXHAUSTION and position.pl > 0.02:
                self.position.close()
                logger.info("Closing position due to FOMO exhaustion")
            
            # Close positions during reversal phase
            elif current_phase == FOMOPhase.REVERSAL and position.pl < -0.01:
                self.position.close()
                logger.info("Closing position due to FOMO reversal")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        current_phase, phase_confidence = self.detect_fomo_phase()
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': current_phase.value,
            'phase_confidence': phase_confidence,
            'fomo_strength': self.fomo_strength[-1],
            'retail_flow': self.retail_flow[-1],
            'social_sentiment': self.social_sentiment[-1],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Phase={current_phase.value}, FOMO={self.fomo_strength[-1]:.3f}")
    
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
