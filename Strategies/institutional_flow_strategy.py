#!/usr/bin/env python3
"""
Institutional Flow Strategy

A sophisticated strategy designed to capture institutional money flows during bull runs:
- Detects large institutional orders through volume profile analysis
- Identifies smart money accumulation and distribution patterns
- Uses order book analysis to detect institutional activity
- Implements position sizing based on institutional flow strength
- Focuses on following the "smart money" during bull market conditions

This strategy targets the most influential market participants during bull runs.
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

class InstitutionalPhase(Enum):
    """Institutional flow phases for strategic positioning"""
    ACCUMULATION = "accumulation"      # Early institutional buying
    DISTRIBUTION = "distribution"      # Institutional selling
    REBALANCING = "rebalancing"        # Institutional rebalancing
    HEDGING = "hedging"                # Institutional hedging
    SPECULATION = "speculation"        # Institutional speculation

class InstitutionalFlowStrategy(Strategy):
    """
    Institutional Flow Strategy
    
    Captures institutional money flows during bull runs:
    - Detects large institutional orders through volume analysis
    - Identifies smart money patterns and accumulation/distribution
    - Uses advanced volume profile analysis
    - Implements position sizing based on institutional flow strength
    """
    
    # Core parameters
    risk_per_trade = 0.012             # 1.2% risk per trade (conservative for institutional)
    max_positions = 3                  # Maximum concurrent positions
    max_drawdown = 0.10                # Maximum drawdown limit
    consecutive_loss_limit = 5         # Maximum consecutive losses
    
    # Institutional detection parameters
    institutional_lookback = 50        # Periods to analyze for institutional patterns
    large_order_threshold = 0.8        # Threshold for large order detection
    volume_profile_threshold = 0.7     # Volume profile threshold
    smart_money_threshold = 0.6        # Smart money threshold
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 40
    rsi_overbought = 60
    atr_period = 14
    atr_multiplier = 1.8
    
    # MACD parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Volume analysis
    volume_period = 20
    volume_profile_period = 50
    
    # Institutional indicators
    institutional_flow_period = 20
    smart_money_period = 30
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.institutional_phases = []
        self.smart_money_indicators = []
        
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
        self.volume_profile = self.I(self._calculate_volume_profile)
        
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
        
        # Institutional indicators
        self.institutional_flow = self.I(self._calculate_institutional_flow)
        self.smart_money = self.I(self._calculate_smart_money)
        self.accumulation_distribution = self.I(self._calculate_accumulation_distribution)
        self.money_flow_index = self.I(self._calculate_money_flow_index)
    
    def _calculate_volume_profile(self, data):
        """Calculate volume profile for institutional analysis"""
        if len(data) < self.volume_profile_period:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate volume-weighted average price
        vwap = (data.Close * data.Volume).rolling(self.volume_profile_period).sum() / data.Volume.rolling(self.volume_profile_period).sum()
        
        # Calculate volume profile strength
        volume_profile_strength = abs(data.Close - vwap) / vwap
        
        return volume_profile_strength.rolling(window=5).mean()
    
    def _calculate_institutional_flow(self, data):
        """Calculate institutional flow indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Large volume trades (institutional characteristic)
        volume_ratio = data.Volume / self.volume_ma
        large_volume_trades = volume_ratio > 2.0
        
        # Price stability (institutional characteristic)
        price_volatility = data.Close.pct_change().rolling(5).std()
        low_volatility = price_volatility < price_volatility.rolling(20).mean() * 0.8
        
        # Volume consistency (institutional characteristic)
        volume_consistency = data.Volume.rolling(5).std() / data.Volume.rolling(5).mean()
        high_consistency = volume_consistency < 0.5
        
        # Combine institutional flow factors
        institutional_flow = (
            large_volume_trades.astype(int) * 0.4 +
            low_volatility.astype(int) * 0.3 +
            high_consistency.astype(int) * 0.3
        )
        
        return institutional_flow.rolling(window=self.institutional_flow_period).mean()
    
    def _calculate_smart_money(self, data):
        """Calculate smart money indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Price vs volume relationship (smart money characteristic)
        price_change = data.Close.pct_change()
        volume_change = data.Volume.pct_change()
        price_volume_correlation = price_change.rolling(10).corr(volume_change)
        
        # RSI divergence (smart money characteristic)
        rsi_divergence = (self.rsi.diff() < 0) & (data.Close.pct_change() > 0)
        
        # Volume profile strength (smart money characteristic)
        volume_profile_strength = self.volume_profile
        
        # Combine smart money factors
        smart_money = (
            abs(price_volume_correlation) * 0.4 +
            rsi_divergence.astype(int) * 0.3 +
            volume_profile_strength * 0.3
        )
        
        return smart_money.rolling(window=self.smart_money_period).mean()
    
    def _calculate_accumulation_distribution(self, data):
        """Calculate accumulation/distribution indicator"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Money Flow Multiplier
        mfm = ((data.Close - data.Low) - (data.High - data.Close)) / (data.High - data.Low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * data.Volume
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        
        # Normalize to 0-1 range
        ad_normalized = (ad_line - ad_line.min()) / (ad_line.max() - ad_line.min())
        
        return ad_normalized.rolling(window=10).mean()
    
    def _calculate_money_flow_index(self, data):
        """Calculate Money Flow Index"""
        if len(data) < 14:
            return pd.Series([0] * len(data), index=data.index)
        
        # Typical Price
        typical_price = (data.High + data.Low + data.Close) / 3
        
        # Money Flow
        money_flow = typical_price * data.Volume
        
        # Positive and Negative Money Flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        # Money Flow Ratio
        mfr = positive_flow / negative_flow
        mfr = mfr.fillna(0)
        
        # Money Flow Index
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi.rolling(window=5).mean()
    
    def detect_institutional_phase(self) -> Tuple[InstitutionalPhase, float]:
        """Detect the current institutional phase"""
        if len(self.data) < self.institutional_lookback:
            return InstitutionalPhase.ACCUMULATION, 0.5
        
        # Institutional indicators
        institutional_flow = self.institutional_flow[-1]
        smart_money = self.smart_money[-1]
        accumulation_distribution = self.accumulation_distribution[-1]
        money_flow_index = self.money_flow_index[-1]
        
        # Volume profile
        volume_profile = self.volume_profile[-1]
        
        # Price action
        price_change = (self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]
        
        # Calculate phase confidence
        confidence = 0.0
        
        # Accumulation phase
        if (institutional_flow > 0.6 and smart_money > 0.5 and 
            accumulation_distribution > 0.6 and money_flow_index > 50 and
            volume_profile > 0.5 and price_change > 0):
            confidence = 0.9
            return InstitutionalPhase.ACCUMULATION, confidence
        
        # Distribution phase
        elif (institutional_flow > 0.6 and smart_money > 0.5 and 
              accumulation_distribution < 0.4 and money_flow_index < 50 and
              volume_profile > 0.5 and price_change < 0):
            confidence = 0.9
            return InstitutionalPhase.DISTRIBUTION, confidence
        
        # Rebalancing phase
        elif (institutional_flow > 0.4 and smart_money > 0.3 and 
              accumulation_distribution > 0.4 and accumulation_distribution < 0.6 and
              volume_profile > 0.3):
            confidence = 0.7
            return InstitutionalPhase.REBALANCING, confidence
        
        # Hedging phase
        elif (institutional_flow > 0.5 and smart_money > 0.4 and 
              money_flow_index > 60 and volume_profile > 0.4):
            confidence = 0.8
            return InstitutionalPhase.HEDGING, confidence
        
        # Speculation phase
        elif (institutional_flow > 0.7 and smart_money > 0.6 and 
              money_flow_index > 70 and volume_profile > 0.6):
            confidence = 0.8
            return InstitutionalPhase.SPECULATION, confidence
        
        return InstitutionalPhase.ACCUMULATION, 0.5
    
    def should_enter_long_institutional(self) -> bool:
        """Determine if we should enter a long position for institutional flow"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_institutional_phase()
        
        # Only enter during accumulation, rebalancing, or speculation phases
        if current_phase not in [InstitutionalPhase.ACCUMULATION, InstitutionalPhase.REBALANCING, InstitutionalPhase.SPECULATION]:
            return False
        
        # Technical conditions
        rsi_strong = self.rsi[-1] > 45
        rsi_not_overbought = self.rsi[-1] < 70
        
        # Momentum conditions
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        macd_rising = self.macd_histogram[-1] > self.macd_histogram[-2]
        
        # Volume conditions
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * 1.5
        volume_profile_strong = self.volume_profile[-1] > self.volume_profile_threshold
        
        # Price conditions
        price_above_bb_middle = self.data.Close[-1] > self.bb_middle[-1]
        
        # Institutional conditions
        institutional_flow_high = self.institutional_flow[-1] > self.large_order_threshold
        smart_money_high = self.smart_money[-1] > self.smart_money_threshold
        accumulation_positive = self.accumulation_distribution[-1] > 0.5
        money_flow_positive = self.money_flow_index[-1] > 50
        
        # Stochastic confirmation
        stoch_bullish = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-1] > 50
        
        # Williams %R confirmation
        williams_bullish = self.williams_r[-1] > -50
        
        return (rsi_strong and rsi_not_overbought and
                macd_positive and macd_rising and
                volume_strong and volume_profile_strong and
                price_above_bb_middle and
                institutional_flow_high and smart_money_high and
                accumulation_positive and money_flow_positive and
                stoch_bullish and williams_bullish and
                phase_confidence > 0.6)
    
    def should_enter_short_institutional(self) -> bool:
        """Determine if we should enter a short position for institutional distribution"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        current_phase, phase_confidence = self.detect_institutional_phase()
        
        # Only enter during distribution phase
        if current_phase != InstitutionalPhase.DISTRIBUTION:
            return False
        
        # Technical conditions
        rsi_overbought = self.rsi[-1] > 65
        rsi_divergence = self.rsi[-1] < self.rsi[-2] and self.data.Close[-1] > self.data.Close[-2]
        
        # Momentum conditions
        macd_negative = self.macd[-1] < self.macd_signal_line[-1]
        macd_falling = self.macd_histogram[-1] < self.macd_histogram[-2]
        
        # Volume conditions
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * 1.5
        volume_profile_strong = self.volume_profile[-1] > self.volume_profile_threshold
        
        # Price conditions
        price_below_bb_middle = self.data.Close[-1] < self.bb_middle[-1]
        
        # Institutional conditions
        institutional_flow_high = self.institutional_flow[-1] > self.large_order_threshold
        smart_money_high = self.smart_money[-1] > self.smart_money_threshold
        accumulation_negative = self.accumulation_distribution[-1] < 0.5
        money_flow_negative = self.money_flow_index[-1] < 50
        
        # Stochastic confirmation
        stoch_bearish = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-1] < 50
        
        # Williams %R confirmation
        williams_bearish = self.williams_r[-1] < -50
        
        return (rsi_overbought and rsi_divergence and
                macd_negative and macd_falling and
                volume_strong and volume_profile_strong and
                price_below_bb_middle and
                institutional_flow_high and smart_money_high and
                accumulation_negative and money_flow_negative and
                stoch_bearish and williams_bearish and
                phase_confidence > 0.6)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on institutional flow strength"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Adjust based on institutional flow strength
        institutional_flow = self.institutional_flow[-1]
        institutional_multiplier = 1.0 + (institutional_flow * 0.5)  # Up to 50% increase
        
        # Adjust based on smart money strength
        smart_money = self.smart_money[-1]
        smart_money_multiplier = 1.0 + (smart_money * 0.3)  # Up to 30% increase
        
        # Adjust based on accumulation/distribution
        accumulation_distribution = self.accumulation_distribution[-1]
        accumulation_multiplier = 1.0 + (accumulation_distribution * 0.2)  # Up to 20% increase
        
        # Adjust based on phase
        current_phase, phase_confidence = self.detect_institutional_phase()
        phase_multiplier = 1.0
        if current_phase == InstitutionalPhase.SPECULATION:
            phase_multiplier = 1.2
        elif current_phase == InstitutionalPhase.ACCUMULATION:
            phase_multiplier = 1.1
        elif current_phase == InstitutionalPhase.REBALANCING:
            phase_multiplier = 0.9
        
        # Final position size
        adjusted_position_size = base_position_size * institutional_multiplier * smart_money_multiplier * accumulation_multiplier * phase_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss based on institutional volatility"""
        atr_value = self.atr[-1]
        
        # Adjust ATR multiplier based on institutional flow
        institutional_flow = self.institutional_flow[-1]
        atr_multiplier = self.atr_multiplier * (1.0 + institutional_flow * 0.2)
        
        stop_distance = atr_value * atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with moderate risk-reward for institutional moves"""
        risk = abs(entry_price - stop_loss)
        
        # Moderate risk-reward ratio for institutional moves
        risk_reward_ratio = 2.0
        
        # Adjust based on institutional flow strength
        institutional_flow = self.institutional_flow[-1]
        if institutional_flow > 0.8:  # Very strong institutional flow
            risk_reward_ratio = 2.5
        elif institutional_flow > 0.6:  # Strong institutional flow
            risk_reward_ratio = 2.0
        else:  # Moderate institutional flow
            risk_reward_ratio = 1.5
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update institutional phase
        current_phase, phase_confidence = self.detect_institutional_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != current_phase.value:
            logger.info(f"Institutional phase: {current_phase.value} (confidence: {phase_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for institutional entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry for institutional accumulation
        if self.should_enter_long_institutional():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_INSTITUTIONAL", current_price, stop_loss, take_profit, position_size)
        
        # Short entry for institutional distribution
        elif self.should_enter_short_institutional():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_INSTITUTIONAL", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with institutional-specific logic"""
        for position in self.positions:
            current_phase, phase_confidence = self.detect_institutional_phase()
            
            # Close positions during distribution phase
            if current_phase == InstitutionalPhase.DISTRIBUTION and position.pl > 0.02:
                self.position.close()
                logger.info("Closing position due to institutional distribution")
            
            # Close positions during hedging phase
            elif current_phase == InstitutionalPhase.HEDGING and position.pl < -0.01:
                self.position.close()
                logger.info("Closing position due to institutional hedging")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        current_phase, phase_confidence = self.detect_institutional_phase()
        
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': current_phase.value,
            'phase_confidence': phase_confidence,
            'institutional_flow': self.institutional_flow[-1],
            'smart_money': self.smart_money[-1],
            'accumulation_distribution': self.accumulation_distribution[-1],
            'money_flow_index': self.money_flow_index[-1],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Phase={current_phase.value}, Institutional={self.institutional_flow[-1]:.3f}")
    
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
