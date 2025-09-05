#!/usr/bin/env python3
"""
Volatility Regime Strategy

A sophisticated strategy that adapts to different volatility environments:
- Low volatility: Mean reversion with tight stops
- High volatility: Breakout trading with wider stops
- Extreme volatility: Defensive positioning
- Dynamic position sizing based on volatility
"""

import pandas as pd
import numpy as np
import talib
import logging
from backtesting import Strategy
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class VolatilityRegimeStrategy(Strategy):
    """
    Volatility Regime Strategy
    
    Adapts trading approach based on volatility environment:
    - Low volatility: Mean reversion strategies
    - Normal volatility: Balanced approach
    - High volatility: Breakout strategies
    - Extreme volatility: Defensive positioning
    """
    
    # Core parameters
    risk_per_trade = 0.01          # 1% risk per trade
    max_positions = 2              # Maximum concurrent positions
    max_drawdown = 0.10            # Maximum drawdown limit (10%)
    consecutive_loss_limit = 3      # Maximum consecutive losses
    
    # Volatility detection
    atr_period = 14
    volatility_lookback = 50       # Periods to analyze volatility
    volatility_smoothing = 5       # Smoothing period for volatility
    
    # Regime thresholds (as multipliers of historical average)
    low_vol_threshold = 0.7        # Below 70% of average = low volatility
    high_vol_threshold = 1.5       # Above 150% of average = high volatility
    extreme_vol_threshold = 2.5    # Above 250% of average = extreme volatility
    
    # Mean reversion parameters
    rsi_period = 14
    rsi_oversold = 25
    rsi_overbought = 75
    
    # Breakout parameters
    bb_period = 20
    bb_std = 2.0
    
    # Moving averages
    ema_fast = 20
    ema_medium = 50
    ema_slow = 200
    
    # Volume analysis
    volume_period = 20
    volume_threshold = 1.2
    
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
        self.current_regime = VolatilityRegime.NORMAL
        self.volatility_ratio = 1.0
        
        # Core indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # Moving averages
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema_50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_medium)
        self.ema_200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        
        # RSI for mean reversion
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # Bollinger Bands for breakout detection
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                                                             timeperiod=self.bb_period,
                                                             nbdevup=self.bb_std,
                                                             nbdevdn=self.bb_std)
        
        # Volume
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
        # Additional volatility indicators
        self.bb_width = self.I(self.calculate_bb_width, self.bb_upper, self.bb_lower, self.bb_middle)
        
        # Keltner Channels for volatility context
        self.kc_upper, self.kc_middle, self.kc_lower = self.I(talib.SAR, self.data.High, self.data.Low,
                                                             acceleration=0.02, maximum=0.2)
        
        logger.info("VolatilityRegimeStrategy initialized successfully")
    
    def calculate_bb_width(self, upper, lower, middle):
        """Calculate Bollinger Band width as volatility measure"""
        return (upper - lower) / middle
    
    def detect_volatility_regime(self) -> tuple[VolatilityRegime, float]:
        """Detect current volatility regime with ratio to historical average"""
        if len(self.data) < self.volatility_lookback:
            return VolatilityRegime.NORMAL, 1.0
        
        # Calculate current volatility (ATR)
        current_atr = self.atr[-1]
        
        # Calculate historical average volatility
        historical_atr = np.mean(self.atr[-self.volatility_lookback:])
        
        # Calculate volatility ratio
        volatility_ratio = current_atr / historical_atr if historical_atr > 0 else 1.0
        
        # Determine regime
        if volatility_ratio < self.low_vol_threshold:
            regime = VolatilityRegime.LOW
        elif volatility_ratio > self.extreme_vol_threshold:
            regime = VolatilityRegime.EXTREME
        elif volatility_ratio > self.high_vol_threshold:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.NORMAL
        
        return regime, volatility_ratio
    
    def get_regime_specific_parameters(self) -> dict:
        """Get trading parameters specific to current volatility regime"""
        base_params = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'entry_threshold': 0.5,
            'max_hold_time': 48,  # hours
            'strategy_type': 'balanced'
        }
        
        if self.current_regime == VolatilityRegime.LOW:
            base_params.update({
                'position_size_multiplier': 1.1,      # Slightly larger positions in low vol
                'stop_loss_multiplier': 0.7,          # Tighter stops for mean reversion
                'take_profit_multiplier': 1.5,        # Lower profit targets
                'entry_threshold': 0.6,               # More selective entry
                'max_hold_time': 24,                  # Shorter holds
                'strategy_type': 'mean_reversion'
            })
        elif self.current_regime == VolatilityRegime.NORMAL:
            base_params.update({
                'position_size_multiplier': 1.0,      # Standard position sizes
                'stop_loss_multiplier': 1.0,          # Standard stops
                'take_profit_multiplier': 2.0,        # Standard profit targets
                'entry_threshold': 0.5,               # Balanced entry
                'max_hold_time': 48,                  # Standard hold time
                'strategy_type': 'balanced'
            })
        elif self.current_regime == VolatilityRegime.HIGH:
            base_params.update({
                'position_size_multiplier': 0.8,      # Smaller positions in high vol
                'stop_loss_multiplier': 1.3,          # Wider stops for breakouts
                'take_profit_multiplier': 2.5,        # Higher profit targets
                'entry_threshold': 0.4,               # More aggressive entry
                'max_hold_time': 72,                  # Longer holds for trends
                'strategy_type': 'breakout'
            })
        else:  # EXTREME
            base_params.update({
                'position_size_multiplier': 0.5,      # Very small positions in extreme vol
                'stop_loss_multiplier': 1.5,          # Very wide stops
                'take_profit_multiplier': 3.0,        # Very high profit targets
                'entry_threshold': 0.3,               # Very aggressive entry
                'max_hold_time': 96,                  # Very long holds
                'strategy_type': 'breakout_aggressive'
            })
        
        return base_params
    
    def should_enter_long(self) -> bool:
        """Determine if we should enter a long position based on volatility regime"""
        if len(self.data) < max(self.ema_slow, self.rsi_period):
            return False
        
        params = self.get_regime_specific_parameters()
        strategy_type = params['strategy_type']
        
        # Base conditions
        price_above_ema = self.data.Close[-1] > self.ema_20[-1]
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        if strategy_type == 'mean_reversion':
            # In low volatility, look for oversold conditions
            return (self.rsi[-1] < self.rsi_oversold and 
                   price_above_ema and 
                   volume_above_average)
        
        elif strategy_type == 'breakout':
            # In high volatility, look for breakouts above resistance
            return (self.data.Close[-1] > self.bb_upper[-1] and 
                   price_above_ema and 
                   volume_above_average * 1.5)
        
        elif strategy_type == 'breakout_aggressive':
            # In extreme volatility, look for strong momentum
            return (self.data.Close[-1] > self.bb_upper[-1] * 1.02 and 
                   price_above_ema and 
                   volume_above_average * 2.0)
        
        else:  # balanced
            # In normal volatility, use balanced approach
            return (price_above_ema and 
                   self.rsi[-1] < 60 and 
                   volume_above_average)
    
    def should_enter_short(self) -> bool:
        """Determine if we should enter a short position based on volatility regime"""
        if len(self.data) < max(self.ema_slow, self.rsi_period):
            return False
        
        params = self.get_regime_specific_parameters()
        strategy_type = params['strategy_type']
        
        # Base conditions
        price_below_ema = self.data.Close[-1] < self.ema_20[-1]
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        if strategy_type == 'mean_reversion':
            # In low volatility, look for overbought conditions
            return (self.rsi[-1] > self.rsi_overbought and 
                   price_below_ema and 
                   volume_above_average)
        
        elif strategy_type == 'breakout':
            # In high volatility, look for breakouts below support
            return (self.data.Close[-1] < self.bb_lower[-1] and 
                   price_below_ema and 
                   volume_above_average * 1.5)
        
        elif strategy_type == 'breakout_aggressive':
            # In extreme volatility, look for strong momentum
            return (self.data.Close[-1] < self.bb_lower[-1] * 0.98 and 
                   price_below_ema and 
                   volume_above_average * 2.0)
        
        else:  # balanced
            # In normal volatility, use balanced approach
            return (price_below_ema and 
                   self.rsi[-1] > 40 and 
                   volume_above_average)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion and volatility adjustment"""
        if stop_loss == entry_price:
            return 0.0
        
        params = self.get_regime_specific_parameters()
        
        # Calculate base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Apply volatility regime multiplier
        adjusted_position_size = base_position_size * params['position_size_multiplier']
        
        # Additional volatility-based adjustment
        if self.current_regime == VolatilityRegime.EXTREME:
            # Further reduce position size in extreme volatility
            adjusted_position_size *= 0.7
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss based on volatility regime"""
        params = self.get_regime_specific_parameters()
        atr_value = self.atr[-1]
        
        # Base stop loss using ATR
        base_stop_distance = atr_value * params['stop_loss_multiplier']
        
        if is_long:
            stop_loss = entry_price - base_stop_distance
            
            # Additional support levels
            if self.current_regime == VolatilityRegime.LOW:
                # In low volatility, use tighter stops
                stop_loss = max(stop_loss, entry_price - (atr_value * 0.5))
            elif self.current_regime == VolatilityRegime.EXTREME:
                # In extreme volatility, use wider stops
                stop_loss = min(stop_loss, entry_price - (atr_value * 3.0))
        else:
            stop_loss = entry_price + base_stop_distance
            
            # Additional resistance levels
            if self.current_regime == VolatilityRegime.LOW:
                # In low volatility, use tighter stops
                stop_loss = min(stop_loss, entry_price + (atr_value * 0.5))
            elif self.current_regime == VolatilityRegime.EXTREME:
                # In extreme volatility, use wider stops
                stop_loss = max(stop_loss, entry_price + (atr_value * 3.0))
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit based on volatility regime and risk-reward"""
        params = self.get_regime_specific_parameters()
        risk = abs(entry_price - stop_loss)
        
        # Get regime-specific risk-reward ratio
        risk_reward_ratio = params['take_profit_multiplier']
        
        if is_long:
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update volatility regime
        self.current_regime, self.volatility_ratio = self.detect_volatility_regime()
        
        # Log regime changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('regime') != self.current_regime.value:
            logger.info(f"Volatility regime changed to: {self.current_regime.value} "
                       f"(ratio: {self.volatility_ratio:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for new entry opportunities based on volatility regime"""
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
        """Manage existing positions with regime-specific logic"""
        for position in self.positions:
            params = self.get_regime_specific_parameters()
            
            # Check if position has been held too long
            if position.pl > 0:  # Profitable position
                # In extreme volatility, consider taking profits earlier
                if (self.current_regime == VolatilityRegime.EXTREME and 
                    position.pl > 0.03):  # 3% profit
                    self.position.close()
                    logger.info(f"Closing profitable position early due to extreme volatility regime")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'regime': self.current_regime.value,
            'volatility_ratio': self.volatility_ratio,
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Regime={self.current_regime.value}, Vol Ratio={self.volatility_ratio:.2f}")
    
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
