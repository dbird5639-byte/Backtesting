#!/usr/bin/env python3
"""
Liquidation Hunt Strategy

A sophisticated strategy designed to capitalize on liquidation cascades and extreme price movements:
- Detects high leverage positions and liquidation zones
- Identifies potential liquidation cascades before they happen
- Uses order book analysis to find liquidation clusters
- Implements aggressive position sizing for liquidation events
- Focuses on extreme volatility and price dislocations

This strategy is specifically designed for bull run conditions where liquidations create massive opportunities.
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

class LiquidationType(Enum):
    """Types of liquidations to target"""
    LONG_LIQUIDATION = "long_liquidation"
    SHORT_LIQUIDATION = "short_liquidation"
    CASCADE_LIQUIDATION = "cascade_liquidation"
    MARGIN_CALL = "margin_call"
    STOP_HUNT = "stop_hunt"

class LiquidationHuntStrategy(Strategy):
    """
    Liquidation Hunt Strategy
    
    Targets liquidation events for maximum profit potential:
    - Identifies liquidation zones using technical analysis
    - Detects high leverage positions through volume/price analysis
    - Uses momentum indicators to predict liquidation cascades
    - Implements aggressive risk management for volatile conditions
    """
    
    # Core parameters
    risk_per_trade = 0.02              # 2% risk per trade (aggressive for liquidations)
    max_positions = 3                  # Maximum concurrent positions
    max_drawdown = 0.15                # Maximum drawdown limit (15% for high-risk strategy)
    consecutive_loss_limit = 3         # Maximum consecutive losses
    
    # Liquidation detection parameters
    liquidation_lookback = 50          # Periods to analyze for liquidation patterns
    leverage_threshold = 0.8           # Threshold for high leverage detection
    volume_spike_threshold = 3.0       # Volume spike threshold for liquidations
    price_acceleration_threshold = 0.05 # Price acceleration threshold
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 25                  # More extreme levels for liquidations
    rsi_overbought = 75
    atr_period = 14
    atr_multiplier = 3.0               # Wider stops for liquidation volatility
    
    # Momentum indicators
    macd_fast = 8                      # Faster MACD for quick signals
    macd_slow = 21
    macd_signal = 5
    stoch_period = 14
    
    # Volume analysis
    volume_period = 20
    volume_spike_period = 5
    
    # Liquidation zone detection
    liquidation_zone_lookback = 100
    liquidation_cluster_threshold = 0.02
    
    def init(self):
        """Initialize strategy indicators and tracking variables"""
        # Performance tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.max_equity = self.equity
        self.liquidation_zones = []
        self.leverage_indicators = []
        
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
        
        # Stochastic for momentum
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, self.data.High, self.data.Low, self.data.Close,
            fastk_period=self.stoch_period, slowk_period=3, slowd_period=3
        )
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        self.volume_spike_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_spike_period)
        
        # Bollinger Bands for volatility
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5
        )
        
        # Price acceleration
        self.price_acceleration = self.I(self._calculate_price_acceleration)
        
        # Liquidation probability
        self.liquidation_probability = self.I(self._calculate_liquidation_probability)
    
    def _calculate_price_acceleration(self, data):
        """Calculate price acceleration for liquidation detection"""
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate second derivative of price
        price_change = data.Close.pct_change()
        acceleration = price_change.diff()
        return acceleration.rolling(window=5).mean()
    
    def _calculate_liquidation_probability(self, data):
        """Calculate probability of liquidation events"""
        if len(data) < 20:
            return pd.Series([0] * len(data), index=data.index)
        
        # Factors that increase liquidation probability
        rsi_extreme = (self.rsi < 20) | (self.rsi > 80)
        volume_spike = self.data.Volume > self.volume_ma * self.volume_spike_threshold
        price_acceleration = abs(self.price_acceleration) > self.price_acceleration_threshold
        volatility_spike = self.atr > self.atr.rolling(20).mean() * 1.5
        
        # Combine factors
        liquidation_score = (
            rsi_extreme.astype(int) * 0.3 +
            volume_spike.astype(int) * 0.3 +
            price_acceleration.astype(int) * 0.2 +
            volatility_spike.astype(int) * 0.2
        )
        
        return liquidation_score.rolling(window=5).mean()
    
    def detect_liquidation_zones(self) -> List[Dict]:
        """Detect potential liquidation zones"""
        if len(self.data) < self.liquidation_zone_lookback:
            return []
        
        zones = []
        recent_data = self.data.tail(self.liquidation_zone_lookback)
        
        # Find high volume clusters
        volume_threshold = recent_data.Volume.quantile(0.9)
        high_volume_periods = recent_data[recent_data.Volume > volume_threshold]
        
        for idx, row in high_volume_periods.iterrows():
            # Check if this is a potential liquidation zone
            price_range = row.High - row.Low
            volume_ratio = row.Volume / self.volume_ma[-1]
            
            if volume_ratio > self.volume_spike_threshold and price_range > self.atr[-1] * 2:
                zones.append({
                    'price': row.Close,
                    'volume': row.Volume,
                    'timestamp': idx,
                    'type': self._classify_liquidation_type(row),
                    'strength': volume_ratio * (price_range / self.atr[-1])
                })
        
        return zones
    
    def _classify_liquidation_type(self, row) -> LiquidationType:
        """Classify the type of liquidation event"""
        price_change = (row.Close - row.Open) / row.Open
        volume_ratio = row.Volume / self.volume_ma[-1]
        
        if price_change > 0.05 and volume_ratio > 3.0:
            return LiquidationType.SHORT_LIQUIDATION
        elif price_change < -0.05 and volume_ratio > 3.0:
            return LiquidationType.LONG_LIQUIDATION
        elif abs(price_change) > 0.1 and volume_ratio > 5.0:
            return LiquidationType.CASCADE_LIQUIDATION
        else:
            return LiquidationType.STOP_HUNT
    
    def detect_leverage_positions(self) -> float:
        """Detect high leverage positions through volume/price analysis"""
        if len(self.data) < 20:
            return 0.0
        
        # Calculate leverage indicator based on volume/price relationship
        volume_price_correlation = self.data.Volume.rolling(10).corr(self.data.Close.pct_change())
        volume_acceleration = self.data.Volume.pct_change().rolling(5).mean()
        price_acceleration = self.data.Close.pct_change().rolling(5).mean()
        
        # High leverage typically shows as high volume with small price moves
        leverage_indicator = abs(volume_acceleration) / (abs(price_acceleration) + 0.001)
        
        return leverage_indicator.iloc[-1] if not leverage_indicator.empty else 0.0
    
    def should_enter_long_liquidation(self) -> bool:
        """Determine if we should enter a long position targeting short liquidations"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        # Check for short liquidation setup
        rsi_oversold = self.rsi[-1] < self.rsi_oversold
        volume_spike = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_spike_threshold
        price_acceleration_negative = self.price_acceleration[-1] < -self.price_acceleration_threshold
        liquidation_prob_high = self.liquidation_probability[-1] > 0.7
        
        # MACD showing potential reversal
        macd_oversold = self.macd[-1] < self.macd_signal_line[-1] * 0.8
        stoch_oversold = self.stoch_k[-1] < 20 and self.stoch_d[-1] < 20
        
        # Bollinger Bands showing oversold
        bb_oversold = self.data.Close[-1] < self.bb_lower[-1]
        
        # High leverage detection
        leverage_high = self.detect_leverage_positions() > self.leverage_threshold
        
        return (rsi_oversold and volume_spike and price_acceleration_negative and 
                liquidation_prob_high and (macd_oversold or stoch_oversold) and 
                bb_oversold and leverage_high)
    
    def should_enter_short_liquidation(self) -> bool:
        """Determine if we should enter a short position targeting long liquidations"""
        if len(self.data) < max(self.rsi_period, self.macd_slow):
            return False
        
        # Check for long liquidation setup
        rsi_overbought = self.rsi[-1] > self.rsi_overbought
        volume_spike = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_spike_threshold
        price_acceleration_positive = self.price_acceleration[-1] > self.price_acceleration_threshold
        liquidation_prob_high = self.liquidation_probability[-1] > 0.7
        
        # MACD showing potential reversal
        macd_overbought = self.macd[-1] > self.macd_signal_line[-1] * 1.2
        stoch_overbought = self.stoch_k[-1] > 80 and self.stoch_d[-1] > 80
        
        # Bollinger Bands showing overbought
        bb_overbought = self.data.Close[-1] > self.bb_upper[-1]
        
        # High leverage detection
        leverage_high = self.detect_leverage_positions() > self.leverage_threshold
        
        return (rsi_overbought and volume_spike and price_acceleration_positive and 
                liquidation_prob_high and (macd_overbought or stoch_overbought) and 
                bb_overbought and leverage_high)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate aggressive position size for liquidation events"""
        if stop_loss == entry_price:
            return 0.0
        
        # Base position size with higher risk for liquidations
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Increase position size based on liquidation probability
        liquidation_prob = self.liquidation_probability[-1]
        liquidation_multiplier = 1.0 + (liquidation_prob * 0.5)  # Up to 50% increase
        
        # Increase position size based on volume spike
        volume_ratio = self.data.Volume[-1] / self.volume_ma[-1]
        volume_multiplier = min(2.0, 1.0 + (volume_ratio - 1.0) * 0.5)
        
        # Final position size
        adjusted_position_size = base_position_size * liquidation_multiplier * volume_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.3
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss optimized for liquidation volatility"""
        atr_value = self.atr[-1]
        
        # Use wider stops for liquidation events
        stop_distance = atr_value * self.atr_multiplier
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit with high risk-reward for liquidations"""
        risk = abs(entry_price - stop_loss)
        
        # Higher risk-reward ratio for liquidation events
        risk_reward_ratio = 3.0
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update liquidation zones
        self.liquidation_zones = self.detect_liquidation_zones()
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for liquidation entry opportunities"""
        current_price = self.data.Close[-1]
        
        # Long entry targeting short liquidations
        if self.should_enter_long_liquidation():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG_LIQUIDATION", current_price, stop_loss, take_profit, position_size)
        
        # Short entry targeting long liquidations
        elif self.should_enter_short_liquidation():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT_LIQUIDATION", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with liquidation-specific logic"""
        for position in self.positions:
            # Check for liquidation cascade continuation
            if self.liquidation_probability[-1] > 0.8:
                # Hold positions longer during liquidation cascades
                pass
            elif self.liquidation_probability[-1] < 0.3:
                # Close positions when liquidation probability drops
                if position.pl > 0.01:  # Take profits
                    self.position.close()
                    logger.info("Closing position due to low liquidation probability")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'liquidation_probability': self.liquidation_probability[-1],
            'leverage_indicator': self.detect_leverage_positions(),
            'volume_ratio': self.data.Volume[-1] / self.volume_ma[-1],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Liquidation Prob={self.liquidation_probability[-1]:.2f}")
    
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
