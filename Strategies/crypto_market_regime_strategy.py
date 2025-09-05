#!/usr/bin/env python3
"""
Crypto Market Regime Strategy

A sophisticated strategy that detects market regimes (bull/bear/sideways) and adapts:
- Market regime detection using multiple indicators
- Dynamic strategy switching based on market conditions
- Bull run momentum capture
- Bear market defensive positioning
- Sideways market mean reversion
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

class MarketRegime(Enum):
    """Market regime enumeration"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    TRANSITION = "transition"

class CryptoMarketRegimeStrategy(Strategy):
    """
    Crypto Market Regime Strategy
    
    Detects market regimes and adapts trading strategy:
    - Bull markets: Trend following with momentum
    - Bear markets: Defensive with short opportunities
    - Sideways markets: Mean reversion with tight stops
    - Transition periods: Reduced position sizes
    """
    
    # Core parameters
    risk_per_trade = 0.008         # 0.8% risk per trade (more conservative)
    max_positions = 3              # Maximum concurrent positions
    max_drawdown = 0.08            # Maximum drawdown limit (8%)
    consecutive_loss_limit = 3      # Maximum consecutive losses
    
    # Market regime detection
    regime_lookback = 50           # Periods to analyze for regime
    regime_threshold = 0.6         # Confidence threshold for regime
    regime_smoothing = 5           # Smoothing period for regime signals
    
    # Trend detection
    ema_fast = 20
    ema_medium = 50
    ema_slow = 200
    
    # Momentum indicators
    rsi_period = 14
    rsi_oversold = 25
    rsi_overbought = 75
    
    # Volatility
    atr_period = 14
    atr_multiplier = 2.0
    
    # Volume analysis
    volume_period = 20
    volume_threshold = 1.5
    
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
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.0
        
        # Core indicators
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema_50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_medium)
        self.ema_200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        
        # Momentum
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # Volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # Volume
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
        # Additional indicators for regime detection
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.data.Close,
                                                            fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Bollinger Bands for volatility context
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                                                             timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        
        # ADX for trend strength
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        logger.info("CryptoMarketRegimeStrategy initialized successfully")
    
    def detect_market_regime(self) -> tuple[MarketRegime, float]:
        """Detect current market regime with confidence score"""
        if len(self.data) < self.regime_lookback:
            return MarketRegime.SIDEWAYS, 0.0
        
        # Get recent data
        recent_close = self.data.Close[-self.regime_lookback:]
        recent_ema_20 = self.ema_20[-self.regime_lookback:]
        recent_ema_50 = self.ema_50[-self.regime_lookback:]
        recent_ema_200 = self.ema_200[-self.regime_lookback:]
        recent_rsi = self.rsi[-self.regime_lookback:]
        recent_adx = self.adx[-self.regime_lookback:]
        recent_macd = self.macd[-self.regime_lookback:]
        recent_volume = self.data.Volume[-self.regime_lookback:]
        
        # Calculate regime scores
        bull_score = 0.0
        bear_score = 0.0
        sideways_score = 0.0
        
        # Trend analysis (40% weight)
        trend_bull = np.sum(recent_close > recent_ema_20) / len(recent_close)
        trend_bull += np.sum(recent_ema_20 > recent_ema_50) / len(recent_close)
        trend_bull += np.sum(recent_ema_50 > recent_ema_200) / len(recent_close)
        trend_bull /= 3.0
        
        trend_bear = 1.0 - trend_bull
        
        # Momentum analysis (30% weight)
        momentum_bull = np.sum(recent_rsi > 50) / len(recent_rsi)
        momentum_bull += np.sum(recent_macd > recent_macd.shift(1)) / len(recent_macd)
        momentum_bull /= 2.0
        
        momentum_bear = 1.0 - momentum_bull
        
        # Volume analysis (20% weight)
        volume_bull = np.sum(recent_volume > self.volume_ma[-self.regime_lookback:]) / len(recent_volume)
        volume_bear = 1.0 - volume_bull
        
        # Trend strength (10% weight)
        avg_adx = np.mean(recent_adx)
        trend_strength = min(avg_adx / 50.0, 1.0)  # Normalize to 0-1
        
        # Calculate final scores
        bull_score = (trend_bull * 0.4 + momentum_bull * 0.3 + volume_bull * 0.2 + trend_strength * 0.1)
        bear_score = (trend_bear * 0.4 + momentum_bear * 0.3 + volume_bear * 0.2 + (1.0 - trend_strength) * 0.1)
        sideways_score = 1.0 - max(bull_score, bear_score)
        
        # Determine regime
        max_score = max(bull_score, bear_score, sideways_score)
        
        if max_score < self.regime_threshold:
            return MarketRegime.TRANSITION, max_score
        
        if bull_score == max_score:
            return MarketRegime.BULL, bull_score
        elif bear_score == max_score:
            return MarketRegime.BEAR, bear_score
        else:
            return MarketRegime.SIDEWAYS, sideways_score
    
    def get_regime_specific_parameters(self) -> dict:
        """Get trading parameters specific to current market regime"""
        base_params = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'entry_threshold': 0.5,
            'max_hold_time': 48  # hours
        }
        
        if self.current_regime == MarketRegime.BULL:
            base_params.update({
                'position_size_multiplier': 1.2,      # Larger positions in bull markets
                'stop_loss_multiplier': 1.5,          # Wider stops for trend following
                'take_profit_multiplier': 2.5,        # Higher profit targets
                'entry_threshold': 0.3,               # More aggressive entry
                'max_hold_time': 72                   # Hold longer in trends
            })
        elif self.current_regime == MarketRegime.BEAR:
            base_params.update({
                'position_size_multiplier': 0.7,      # Smaller positions in bear markets
                'stop_loss_multiplier': 0.8,          # Tighter stops
                'take_profit_multiplier': 1.5,        # Lower profit targets
                'entry_threshold': 0.7,               # More conservative entry
                'max_hold_time': 24                   # Shorter holds
            })
        elif self.current_regime == MarketRegime.SIDEWAYS:
            base_params.update({
                'position_size_multiplier': 0.8,      # Moderate position sizes
                'stop_loss_multiplier': 0.9,          # Moderate stops
                'take_profit_multiplier': 1.8,        # Moderate profit targets
                'entry_threshold': 0.6,               # Balanced entry
                'max_hold_time': 36                   # Moderate hold time
            })
        else:  # TRANSITION
            base_params.update({
                'position_size_multiplier': 0.5,      # Small positions during transitions
                'stop_loss_multiplier': 0.7,          # Very tight stops
                'take_profit_multiplier': 1.2,        # Quick profits
                'entry_threshold': 0.8,               # Very conservative entry
                'max_hold_time': 12                   # Very short holds
            })
        
        return base_params
    
    def should_enter_long(self) -> bool:
        """Determine if we should enter a long position"""
        if len(self.data) < max(self.ema_slow, self.rsi_period):
            return False
        
        params = self.get_regime_specific_parameters()
        
        # Base conditions
        price_above_ema = self.data.Close[-1] > self.ema_20[-1]
        ema_trending_up = self.ema_20[-1] > self.ema_20[-2]
        rsi_not_overbought = self.rsi[-1] < self.rsi_overbought
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        # Regime-specific conditions
        if self.current_regime == MarketRegime.BULL:
            # In bull markets, be more aggressive
            return (price_above_ema and ema_trending_up and rsi_not_overbought and volume_above_average)
        
        elif self.current_regime == MarketRegime.BEAR:
            # In bear markets, only take strong reversal signals
            return (price_above_ema and self.rsi[-1] < 40 and volume_above_average * 1.5)
        
        elif self.current_regime == MarketRegime.SIDEWAYS:
            # In sideways markets, look for mean reversion
            return (self.rsi[-1] < 35 and price_above_ema and volume_above_average)
        
        else:  # TRANSITION
            # During transitions, be very selective
            return (price_above_ema and self.rsi[-1] < 30 and volume_above_average * 2.0)
    
    def should_enter_short(self) -> bool:
        """Determine if we should enter a short position"""
        if len(self.data) < max(self.ema_slow, self.rsi_period):
            return False
        
        params = self.get_regime_specific_parameters()
        
        # Base conditions
        price_below_ema = self.data.Close[-1] < self.ema_20[-1]
        ema_trending_down = self.ema_20[-1] < self.ema_20[-2]
        rsi_not_oversold = self.rsi[-1] > self.rsi_oversold
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        # Regime-specific conditions
        if self.current_regime == MarketRegime.BULL:
            # In bull markets, avoid shorts
            return False
        
        elif self.current_regime == MarketRegime.BEAR:
            # In bear markets, be aggressive with shorts
            return (price_below_ema and ema_trending_down and rsi_not_oversold and volume_above_average)
        
        elif self.current_regime == MarketRegime.SIDEWAYS:
            # In sideways markets, look for mean reversion
            return (self.rsi[-1] > 65 and price_below_ema and volume_above_average)
        
        else:  # TRANSITION
            # During transitions, be very selective
            return (price_below_ema and self.rsi[-1] > 70 and volume_above_average * 2.0)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion and regime adjustment"""
        if stop_loss == entry_price:
            return 0.0
        
        params = self.get_regime_specific_parameters()
        
        # Calculate base position size
        risk_amount = self.equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Apply regime-specific multiplier
        adjusted_position_size = base_position_size * params['position_size_multiplier']
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss based on regime and volatility"""
        params = self.get_regime_specific_parameters()
        atr_value = self.atr[-1]
        
        if is_long:
            stop_loss = entry_price - (atr_value * self.atr_multiplier * params['stop_loss_multiplier'])
        else:
            stop_loss = entry_price + (atr_value * self.atr_multiplier * params['stop_loss_multiplier'])
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit based on regime and risk-reward"""
        params = self.get_regime_specific_parameters()
        risk = abs(entry_price - stop_loss)
        
        if is_long:
            take_profit = entry_price + (risk * params['take_profit_multiplier'])
        else:
            take_profit = entry_price - (risk * params['take_profit_multiplier'])
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update market regime
        self.current_regime, self.regime_confidence = self.detect_market_regime()
        
        # Log regime changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('regime') != self.current_regime.value:
            logger.info(f"Market regime changed to: {self.current_regime.value} (confidence: {self.regime_confidence:.2f})")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for new entry opportunities based on current regime"""
        current_price = self.data.Close[-1]
        
        # Long entry
        if self.should_enter_long():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG", current_price, stop_loss, take_profit, position_size)
        
        # Short entry (only in bear or sideways markets)
        elif self.current_regime in [MarketRegime.BEAR, MarketRegime.SIDEWAYS] and self.should_enter_short():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT", current_price, stop_loss, take_profit, position_size)
    
    def manage_existing_positions(self):
        """Manage existing positions with regime-specific logic"""
        for position in self.positions:
            # Check if position has been held too long
            params = self.get_regime_specific_parameters()
            if position.pl > 0:  # Profitable position
                # Consider taking partial profits in certain regimes
                if self.current_regime == MarketRegime.TRANSITION and position.pl > 0.02:
                    self.position.close()
                    logger.info(f"Closing profitable position early due to transition regime")
    
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
            'regime_confidence': self.regime_confidence,
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Regime={self.current_regime.value}")
    
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
