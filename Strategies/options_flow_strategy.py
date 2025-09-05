#!/usr/bin/env python3
"""
Options Flow Strategy

A sophisticated strategy designed to capitalize on options flow data during bull runs:
- Detects unusual options activity and flow patterns
- Identifies gamma squeezes and options-driven moves
- Uses options Greeks simulation for market dynamics
- Implements position sizing based on options flow strength
- Focuses on capturing options-driven price movements

This strategy targets the most influential options activity during bull run conditions.
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

class OptionsFlowPhase(Enum):
    """Options flow phases for strategic positioning"""
    ACCUMULATION = "accumulation"      # Options accumulation
    GAMMA_SQUEEZE = "gamma_squeeze"    # Gamma squeeze conditions
    EXPIRATION = "expiration"          # Options expiration
    HEDGING = "hedging"                # Options hedging activity
    SPECULATION = "speculation"        # Options speculation

class OptionsFlowStrategy(Strategy):
    """
    Options Flow Strategy
    
    Captures options-driven moves during bull runs:
    - Detects unusual options activity and flow patterns
    - Identifies gamma squeezes and options-driven moves
    - Uses options Greeks simulation for market dynamics
    - Implements position sizing based on options flow strength
    """
    
    # Core parameters
    risk_per_trade = 0.014             # 1.4% risk per trade
    max_positions = 3                  # Maximum concurrent positions
    max_drawdown = 0.11                # Maximum drawdown limit
    consecutive_loss_limit = 4         # Maximum consecutive losses
    
    # Options flow detection parameters
    options_lookback = 30              # Periods to analyze for options patterns
    unusual_activity_threshold = 0.7   # Unusual activity threshold
    gamma_squeeze_threshold = 0.8      # Gamma squeeze threshold
    put_call_ratio_threshold = 0.6     # Put/call ratio threshold
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 35
    rsi_overbought = 65
    atr_period = 14
    atr_multiplier = 2.2
    
    # MACD parameters
    macd_fast = 8
    macd_slow = 21
    macd_signal = 5
    
    # Volume analysis
    volume_period = 20
    volume_options_threshold = 1.8
    
    # Options indicators
    options_flow_period = 15
    gamma_period = 10
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.options_phases = []
        self.gamma_events = []
        
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
        
        # Options indicators
        self.options_flow = self.I(self._calculate_options_flow)
        self.put_call_ratio = self.I(self._calculate_put_call_ratio)
        self.gamma_squeeze = self.I(self._calculate_gamma_squeeze)
        self.unusual_activity = self.I(self._calculate_unusual_activity)
        self.options_momentum = self.I(self._calculate_options_momentum)
    
    def _calculate_options_flow(self, data):
        """Calculate options flow indicator (simulated)"""
        if len(data) < 10:
            return pd.Series([0.5] * len(data), index=data.index)
        
        # Simulate options flow based on price action and volume
        price_momentum = data.Close.pct_change(5)
        volume_momentum = data.Volume.pct_change(5)
        
        # Options flow increases with positive momentum and volume
        options_flow = (
            price_momentum * 0.6 +
            volume_momentum * 0.4
        )
        
        # Normalize to 0-1 range
        options_flow = (options_flow - options_flow.min()) / (options_flow.max() - options_flow.min())
        options_flow = options_flow.fillna(0.5)
        
        return options_flow.rolling(window=self.options_flow_period).mean()
    
    def _calculate_put_call_ratio(self, data):
        """Calculate put/call ratio indicator (simulated)"""
        if len(data) < 10:
            return pd.Series([0.5] * len(data), index=data.index)
        
        # Simulate put/call ratio based on price action
        price_change = data.Close.pct_change(3)
        
        # Put/call ratio increases with negative price changes
        put_call_ratio = 0.5 - (price_change * 2)
        put_call_ratio = np.clip(put_call_ratio, 0, 1)
        
        return put_call_ratio.rolling(window=5).mean()
    
    def _calculate_gamma_squeeze(self, data):
        """Calculate gamma squeeze indicator (simulated)"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Simulate gamma squeeze based on volatility and price action
        volatility = data.Close.pct_change().rolling(5).std()
        price_momentum = data.Close.pct_change(3)
        volume_surge = data.Volume / self.volume_ma
        
        # Gamma squeeze conditions
        gamma_squeeze = (
            (volatility < volatility.rolling(20).mean() * 0.8).astype(int) * 0.4 +
            (abs(price_momentum) > 0.02).astype(int) * 0.3 +
            (volume_surge > 1.5).astype(int) * 0.3
        )
        
        return gamma_squeeze.rolling(window=self.gamma_period).mean()
    
    def _calculate_unusual_activity(self, data):
        """Calculate unusual options activity indicator (simulated)"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Simulate unusual activity based on volume and price action
        volume_ratio = data.Volume / self.volume_ma
        price_volatility = data.Close.pct_change().rolling(3).std()
        
        # Unusual activity increases with high volume and volatility
        unusual_activity = (
            (volume_ratio > 2.0).astype(int) * 0.5 +
            (price_volatility > price_volatility.rolling(20).mean() * 1.5).astype(int) * 0.5
        )
        
        return unusual_activity.rolling(window=5).mean()
    
    def _calculate_options_momentum(self, data):
        """Calculate options momentum indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Options momentum based on flow and activity
        flow_momentum = self.options_flow.diff(3)
        activity_momentum = self.unusual_activity.diff(3)
        gamma_momentum = self.gamma_squeeze.diff(3)
        
        # Combine momentum factors
        options_momentum = (
            flow_momentum * 0.4 +
            activity_momentum * 0.3 +
            gamma_momentum * 0.3
        )
        
        return options_momentum.rolling(window=5).mean()
    
    def detect_options_phase(self) -> Tuple[OptionsFlowPhase, float]:
        """Detect the current options flow phase"""
        if len(self.data) < self.options_lookback:
            return OptionsFlowPhase.ACCUMULATION, 0.5
        
        # Options indicators
        options_flow = self.options_flow[-1]
        put_call_ratio = self.put_call_ratio[-1]
        gamma_squeeze = self.gamma_squeeze[-1]
        unusual_activity = self.unusual_activity[-1]
        options_momentum = self.options_momentum[-1]
        
        # Volume surge
        volume_surge = self.data.Volume[-1] / self.volume_ma[-1]
        
        # Price action
        price_change = (self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]
        
        # Calculate phase confidence
        confidence = 0.0
        
        # Accumulation phase
        if (options_flow > 0.6 and put_call_ratio < 0.4 and 
            unusual_activity > 0.5 and options_momentum > 0.1):
            confidence = 0.9
            return OptionsFlowPhase.ACCUMULATION, confidence
        
        # Gamma squeeze phase
        elif (gamma_squeeze > self.gamma_squeeze_threshold and 
              unusual_activity > 0.7 and volume_surge > 2.0 and
              abs(price_change) > 0.02):
            confidence = 0.95
            return OptionsFlowPhase.GAMMA_SQUEEZE, confidence
        
        # Expiration phase
        elif (options_flow > 0.8 and put_call_ratio > 0.6 and 
              unusual_activity > 0.8 and volume_surge > 2.5):
            confidence = 0.9
            return OptionsFlowPhase.EXPIRATION, confidence
        
        # Hedging phase
        elif (options_flow > 0.5 and put_call_ratio > 0.5 and 
              unusual_activity > 0.6 and options_momentum < -0.1):
            confidence = 0.8
            return OptionsFlowPhase.HEDGING, confidence
        
        # Speculation phase
        elif (options_flow > 0.7 and put_call_ratio < 0.3 and 
              unusual_activity > 0.6 and options_momentum > 0.2):
            confidence = 0.8
            return OptionsFlowPhase.SPECULATION, confidence
        
        return OptionsFlowPhase.ACCUMULATION, 0.5
    
    def should_enter_long_options(self) -> bool:
        """Determine if we should enter a long position for options flow"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_options_phase()
        
        # Only enter during accumulation, gamma squeeze, or speculation phases
        if current_phase not in [OptionsFlowPhase.ACCUMULATION, OptionsFlowPhase.GAMMA_SQUEEZE, OptionsFlowPhase.SPECULATION]:
            return False
        
        # Technical conditions
        rsi_strong = self.rsi[-1] > 45
        rsi_not_overbought = self.rsi[-1] < 75
        
        # Momentum conditions
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        macd_rising = self.macd_histogram[-1] > self.macd_histogram[-2]
        
        # Volume conditions
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_options_threshold
        
        # Price conditions
        price_above_bb_middle = self.data.Close[-1] > self.bb_middle[-1]
        
        # Options conditions
        options_flow_high = self.options_flow[-1] > self.unusual_activity_threshold
        put_call_ratio_low = self.put_call_ratio[-1] < self.put_call_ratio_threshold
        unusual_activity_high = self.unusual_activity[-1] > 0.6
        options_momentum_positive = self.options_momentum[-1] > 0.1
        
        # Stochastic confirmation
        stoch_bullish = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-1] > 50
        
        # Williams %R confirmation
        williams_bullish = self.williams_r[-1] > -50
        
        return (rsi_strong and rsi_not_overbought and
                macd_positive and macd_rising and
                volume_strong and
                price_above_bb_middle and
                options_flow_high and put_call_ratio_low and
                unusual_activity_high and options_momentum_positive and
                stoch_bullish and williams_bullish and
                phase_confidence > 0.6)
    
    def should_enter_short_options(self) -> bool:
        """Determine if we should enter a short position for options flow"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_options_phase()
        
        # Only enter during expiration or hedging phases
        if current_phase not in [OptionsFlowPhase.EXPIRATION, OptionsFlowPhase.HEDGING]:
            return False
        
        # Technical conditions
        rsi_overbought = self.rsi[-1] > 70
        rsi_divergence = self.rsi[-1] < self.rsi[-2] and self.data.Close[-1] > self.data.Close[-2]
        
        # Momentum conditions
        macd_negative = self.macd[-1] < self.macd_signal_line[-1]
        macd_falling = self.macd_histogram[-1] < self.macd_histogram[-2]
        
        # Volume conditions
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_options_threshold
        
        # Price conditions
        price_below_bb_middle = self.data.Close[-1] < self.bb_middle[-1]
        
        # Options conditions
        options_flow_high = self.options_flow[-1] > self.unusual_activity_threshold
        put_call_ratio_high = self.put_call_ratio[-1] > 0.6
        unusual_activity_high = self.unusual_activity[-1] > 0.6
        options_momentum_negative = self.options_momentum[-1] < -0.1
        
        # Stochastic confirmation
        stoch_bearish = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-1] < 50
        
        # Williams %R confirmation
        williams_bearish = self.williams_r[-1] < -50
        
        return (rsi_overbought and rsi_divergence and
                macd_negative and macd_falling and
                volume_strong and
                price_below_bb_middle and
                options_flow_high and put_call_ratio_high and
                unusual_activity_high and options_momentum_negative and
                stoch_bearish and williams_bearish and
                phase_confidence > 0.6)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on options flow strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust based on options flow strength
        options_flow = self.options_flow[-1]
        options_multiplier = 1.0 + (options_flow * 0.4)  # Up to 40% increase
        
        # Adjust based on unusual activity
        unusual_activity = self.unusual_activity[-1]
        activity_multiplier = 1.0 + (unusual_activity * 0.3)  # Up to 30% increase
        
        # Adjust based on gamma squeeze
        gamma_squeeze = self.gamma_squeeze[-1]
        gamma_multiplier = 1.0 + (gamma_squeeze * 0.5)  # Up to 50% increase
        
        # Adjust based on phase
        current_phase, phase_confidence = self.detect_options_phase()
        phase_multiplier = 1.0
        if current_phase == OptionsFlowPhase.GAMMA_SQUEEZE:
            phase_multiplier = 1.5
        elif current_phase == OptionsFlowPhase.ACCUMULATION:
            phase_multiplier = 1.2
        elif current_phase == OptionsFlowPhase.SPECULATION:
            phase_multiplier = 1.1
        
        # Final position size
        adjusted_position_size = base_position_size * options_multiplier * activity_multiplier * gamma_multiplier * phase_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss based on options volatility"""
        atr_value = self.atr[-1]
        
        # Adjust ATR multiplier based on options activity
        unusual_activity = self.unusual_activity[-1]
        atr_multiplier = self.atr_multiplier * (1.0 + unusual_activity * 0.3)
        
        stop_distance = atr_value * atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with high risk-reward for options moves"""
        risk = abs(entry_price - stop_loss)
        
        # Higher risk-reward ratio for options moves
        risk_reward_ratio = 2.5
        
        # Adjust based on gamma squeeze
        gamma_squeeze = self.gamma_squeeze[-1]
        if gamma_squeeze > 0.8:  # Very strong gamma squeeze
            risk_reward_ratio = 3.0
        elif gamma_squeeze > 0.6:  # Strong gamma squeeze
            risk_reward_ratio = 2.5
        else:  # Moderate options activity
            risk_reward_ratio = 2.0
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update options phase
        current_phase, phase_confidence = self.detect_options_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != current_phase.value:
            logger.info(f"Options phase: {current_phase.value} (confidence: {phase_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for options entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry for options flow
        if self.should_enter_long_options():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_OPTIONS", current_price, stop_loss, take_profit, position_size)
        
        # Short entry for options flow
        elif self.should_enter_short_options():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_OPTIONS", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with options-specific logic"""
        for position in self.positions:
            current_phase, phase_confidence = self.detect_options_phase()
            
            # Close positions during expiration phase
            if current_phase == OptionsFlowPhase.EXPIRATION and position.pl > 0.02:
                self.position.close()
                logger.info("Closing position due to options expiration")
            
            # Close positions during hedging phase
            elif current_phase == OptionsFlowPhase.HEDGING and position.pl < -0.01:
                self.position.close()
                logger.info("Closing position due to options hedging")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        current_phase, phase_confidence = self.detect_options_phase()
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': current_phase.value,
            'phase_confidence': phase_confidence,
            'options_flow': self.options_flow[-1],
            'put_call_ratio': self.put_call_ratio[-1],
            'gamma_squeeze': self.gamma_squeeze[-1],
            'unusual_activity': self.unusual_activity[-1],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Phase={current_phase.value}, Options={self.options_flow[-1]:.3f}")
    
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
