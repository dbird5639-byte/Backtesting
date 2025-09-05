#!/usr/bin/env python3
"""
Smart Risk Management Strategy

A sophisticated strategy focused on advanced risk management:
- Dynamic position sizing using Kelly Criterion and volatility
- Portfolio heat management and correlation analysis
- Dynamic risk adjustment based on market conditions
- Advanced stop-loss and take-profit mechanisms
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

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class SmartRiskManagementStrategy(Strategy):
    """
    Smart Risk Management Strategy
    
    Advanced risk management features:
    - Dynamic position sizing using Kelly Criterion
    - Portfolio heat management
    - Correlation-based risk adjustment
    - Advanced stop-loss mechanisms
    - Dynamic risk-reward optimization
    """
    
    # Core risk parameters
    base_risk_per_trade = 0.01     # 1% base risk per trade
    max_portfolio_risk = 0.05      # 5% maximum portfolio risk
    max_positions = 3              # Maximum concurrent positions
    max_drawdown = 0.08            # Maximum drawdown limit (8%)
    consecutive_loss_limit = 3      # Maximum consecutive losses
    
    # Risk adjustment parameters
    volatility_risk_multiplier = 1.5
    correlation_risk_multiplier = 1.3
    market_regime_risk_multiplier = 1.2
    
    # Kelly Criterion parameters
    kelly_fraction = 0.25          # Conservative Kelly fraction
    min_kelly_multiplier = 0.1     # Minimum position size multiplier
    max_kelly_multiplier = 2.0     # Maximum position size multiplier
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
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
        self.current_risk_level = RiskLevel.MEDIUM
        self.portfolio_heat = 0.0
        
        # Core indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # Moving averages
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema_50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_medium)
        self.ema_200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        
        # Volume
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
        # Additional risk indicators
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                                                             timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        
        # MACD for trend confirmation
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.data.Close,
                                                            fastperiod=12, slowperiod=26, signalperiod=9)
        
        logger.info("SmartRiskManagementStrategy initialized successfully")
    
    def calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        if not self.positions:
            return 0.0
        
        total_risk = 0.0
        for position in self.positions:
            # Calculate risk for each position
            entry_price = position.entry_price
            stop_loss = position.sl
            position_size = position.size
            
            if stop_loss and entry_price != stop_loss:
                price_risk = abs(entry_price - stop_loss)
                position_risk = (price_risk * position_size) / self.equity
                total_risk += position_risk
        
        return total_risk
    
    def assess_market_risk(self) -> RiskLevel:
        """Assess current market risk level based on multiple factors"""
        if len(self.data) < 50:
            return RiskLevel.MEDIUM
        
        # Volatility risk
        current_atr = self.atr[-1]
        avg_atr = np.mean(self.atr[-20:])
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Trend risk
        price = self.data.Close[-1]
        ema_20 = self.ema_20[-1]
        ema_50 = self.ema_50[-1]
        ema_200 = self.ema_200[-1]
        
        # Calculate trend strength and direction
        trend_strength = 0.0
        if ema_20 > ema_50 > ema_200:
            trend_strength = 1.0  # Strong uptrend
        elif ema_20 < ema_50 < ema_200:
            trend_strength = -1.0  # Strong downtrend
        elif ema_20 > ema_50:
            trend_strength = 0.5  # Moderate uptrend
        elif ema_20 < ema_50:
            trend_strength = -0.5  # Moderate downtrend
        
        # Volume risk
        current_volume = self.data.Volume[-1]
        avg_volume = self.volume_ma[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # RSI risk
        current_rsi = self.rsi[-1]
        rsi_risk = 0.0
        if current_rsi < 20 or current_rsi > 80:
            rsi_risk = 1.0  # Extreme overbought/oversold
        elif current_rsi < 30 or current_rsi > 70:
            rsi_risk = 0.5  # Moderate overbought/oversold
        
        # Calculate overall risk score
        risk_score = 0.0
        risk_score += (volatility_ratio - 1.0) * 0.3  # Volatility contribution
        risk_score += abs(trend_strength) * 0.2        # Trend contribution
        risk_score += (volume_ratio - 1.0) * 0.2      # Volume contribution
        risk_score += rsi_risk * 0.3                   # RSI contribution
        
        # Normalize risk score to 0-1
        risk_score = max(0.0, min(1.0, (risk_score + 1.0) / 2.0))
        
        # Determine risk level
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where: b = odds received, p = probability of win, q = probability of loss
        b = avg_win / avg_loss  # Risk-reward ratio
        p = win_rate            # Probability of win
        q = 1 - win_rate        # Probability of loss
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative Kelly fraction
        conservative_kelly = kelly_fraction * self.kelly_fraction
        
        # Constrain to reasonable bounds
        constrained_kelly = max(self.min_kelly_multiplier, 
                              min(self.max_kelly_multiplier, conservative_kelly))
        
        return constrained_kelly
    
    def calculate_dynamic_risk_per_trade(self) -> float:
        """Calculate dynamic risk per trade based on current conditions"""
        base_risk = self.base_risk_per_trade
        
        # Adjust based on market risk level
        market_risk = self.assess_market_risk()
        if market_risk == RiskLevel.LOW:
            risk_multiplier = 1.2  # Increase risk in low-risk environments
        elif market_risk == RiskLevel.MEDIUM:
            risk_multiplier = 1.0  # Standard risk
        elif market_risk == RiskLevel.HIGH:
            risk_multiplier = 0.7  # Reduce risk in high-risk environments
        else:  # EXTREME
            risk_multiplier = 0.5  # Significantly reduce risk in extreme environments
        
        # Adjust based on portfolio heat
        portfolio_heat = self.calculate_portfolio_heat()
        if portfolio_heat > self.max_portfolio_risk * 0.8:
            heat_multiplier = 0.5  # Reduce risk when portfolio is hot
        elif portfolio_heat > self.max_portfolio_risk * 0.5:
            heat_multiplier = 0.7  # Moderate risk reduction
        else:
            heat_multiplier = 1.0  # No heat-based reduction
        
        # Adjust based on consecutive losses
        loss_multiplier = 1.0
        if self.consecutive_losses >= 2:
            loss_multiplier = 0.7  # Reduce risk after consecutive losses
        
        # Calculate final risk per trade
        final_risk = base_risk * risk_multiplier * heat_multiplier * loss_multiplier
        
        # Ensure risk stays within reasonable bounds
        final_risk = max(0.002, min(0.02, final_risk))  # Between 0.2% and 2%
        
        return final_risk
    
    def should_enter_long(self) -> bool:
        """Determine if we should enter a long position"""
        if len(self.data) < max(self.ema_slow, self.rsi_period):
            return False
        
        # Check portfolio heat
        portfolio_heat = self.calculate_portfolio_heat()
        if portfolio_heat >= self.max_portfolio_risk:
            return False
        
        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Technical conditions
        current_price = self.data.Close[-1]
        price_above_ema = current_price > self.ema_20[-1]
        ema_trending_up = self.ema_20[-1] > self.ema_20[-2]
        rsi_not_overbought = self.rsi[-1] < self.rsi_overbought
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        # MACD confirmation
        macd_bullish = self.macd[-1] > self.macd_signal[-1]
        
        return (price_above_ema and ema_trending_up and rsi_not_overbought and 
                volume_above_average and macd_bullish)
    
    def should_enter_short(self) -> bool:
        """Determine if we should enter a short position"""
        if len(self.data) < max(self.ema_slow, self.rsi_period):
            return False
        
        # Check portfolio heat
        portfolio_heat = self.calculate_portfolio_heat()
        if portfolio_heat >= self.max_portfolio_risk:
            return False
        
        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Technical conditions
        current_price = self.data.Close[-1]
        price_below_ema = current_price < self.ema_20[-1]
        ema_trending_down = self.ema_20[-1] < self.ema_20[-2]
        rsi_not_oversold = self.rsi[-1] > self.rsi_oversold
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        # MACD confirmation
        macd_bearish = self.macd[-1] < self.macd_signal[-1]
        
        return (price_below_ema and ema_trending_down and rsi_not_oversold and 
                volume_above_average and macd_bearish)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using dynamic risk management"""
        if stop_loss == entry_price:
            return 0.0
        
        # Get dynamic risk per trade
        risk_per_trade = self.calculate_dynamic_risk_per_trade()
        
        # Calculate base position size
        risk_amount = self.equity * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Apply Kelly Criterion if we have historical data
        if len(self.trades_history) >= 10:
            # Calculate historical performance metrics
            recent_trades = self.trades_history[-10:]
            wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
            losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
            
            if wins and losses:
                win_rate = len(wins) / len(recent_trades)
                avg_win = np.mean([t.get('pnl', 0) for t in wins])
                avg_loss = abs(np.mean([t.get('pnl', 0) for t in losses]))
                
                kelly_multiplier = self.calculate_kelly_position_size(win_rate, avg_win, avg_loss)
                base_position_size *= kelly_multiplier
        
        # Apply volatility adjustment
        current_atr = self.atr[-1]
        avg_atr = np.mean(self.atr[-20:])
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        if volatility_ratio > 1.5:
            # High volatility - reduce position size
            base_position_size *= 0.7
        elif volatility_ratio < 0.7:
            # Low volatility - increase position size slightly
            base_position_size *= 1.1
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            base_position_size *= 0.5
        
        return base_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss based on volatility and support/resistance"""
        atr_value = self.atr[-1]
        
        # Base stop loss using ATR
        base_stop_distance = atr_value * self.atr_multiplier
        
        # Adjust based on market risk level
        market_risk = self.assess_market_risk()
        if market_risk == RiskLevel.LOW:
            stop_multiplier = 0.8  # Tighter stops in low-risk environments
        elif market_risk == RiskLevel.MEDIUM:
            stop_multiplier = 1.0  # Standard stops
        elif market_risk == RiskLevel.HIGH:
            stop_multiplier = 1.2  # Wider stops in high-risk environments
        else:  # EXTREME
            stop_multiplier = 1.5  # Much wider stops in extreme environments
        
        final_stop_distance = base_stop_distance * stop_multiplier
        
        if is_long:
            stop_loss = entry_price - final_stop_distance
            
            # Additional support levels
            if self.ema_50[-1] < entry_price:
                stop_loss = max(stop_loss, self.ema_50[-1] * 0.995)
        else:
            stop_loss = entry_price + final_stop_distance
            
            # Additional resistance levels
            if self.ema_50[-1] > entry_price:
                stop_loss = min(stop_loss, self.ema_50[-1] * 1.005)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit based on dynamic risk-reward optimization"""
        risk = abs(entry_price - stop_loss)
        
        # Base risk-reward ratio
        base_risk_reward = 2.0
        
        # Adjust based on market risk level
        market_risk = self.assess_market_risk()
        if market_risk == RiskLevel.LOW:
            risk_reward_multiplier = 1.5  # Lower targets in low-risk environments
        elif market_risk == RiskLevel.MEDIUM:
            risk_reward_multiplier = 2.0  # Standard targets
        elif market_risk == RiskLevel.HIGH:
            risk_reward_multiplier = 2.5  # Higher targets in high-risk environments
        else:  # EXTREME
            risk_reward_multiplier = 3.0  # Much higher targets in extreme environments
        
        final_risk_reward = base_risk_reward * risk_reward_multiplier
        
        if is_long:
            take_profit = entry_price + (risk * final_risk_reward)
        else:
            take_profit = entry_price - (risk * final_risk_reward)
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update risk assessments
        self.current_risk_level = self.assess_market_risk()
        self.portfolio_heat = self.calculate_portfolio_heat()
        
        # Log significant changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('risk_level') != self.current_risk_level.value:
            logger.info(f"Risk level changed to: {self.current_risk_level.value}, "
                       f"Portfolio heat: {self.portfolio_heat:.3f}")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for new entry opportunities with risk management"""
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
        """Manage existing positions with advanced risk management"""
        for position in self.positions:
            # Check portfolio heat and close positions if necessary
            portfolio_heat = self.calculate_portfolio_heat()
            if portfolio_heat > self.max_portfolio_risk:
                # Close worst performing position
                if position.pl < 0:
                    self.position.close()
                    logger.info(f"Closing losing position due to portfolio heat limit: {portfolio_heat:.3f}")
            
            # Implement trailing stops for profitable positions
            if position.pl > 0:
                entry_price = position.entry_price
                current_price = self.data.Close[-1]
                
                if position.is_long:
                    risk = entry_price - position.sl
                    if current_price > entry_price + risk:
                        # Move stop to breakeven + buffer
                        new_stop = entry_price + (risk * 0.1)
                        if new_stop > position.sl:
                            position.sl = new_stop
                else:
                    risk = position.sl - entry_price
                    if current_price < entry_price - risk:
                        # Move stop to breakeven + buffer
                        new_stop = entry_price - (risk * 0.1)
                        if new_stop < position.sl:
                            position.sl = new_stop
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float):
        """Record trade details for analysis"""
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_level': self.current_risk_level.value,
            'portfolio_heat': self.portfolio_heat,
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Risk Level={self.current_risk_level.value}, "
                   f"Portfolio Heat={self.portfolio_heat:.3f}")
    
    def on_trade_exit(self, position, exit_price, exit_time):
        """Handle trade exit events"""
        # Update consecutive losses
        if position.pl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Record trade result for analysis
        if self.trades_history:
            last_trade = self.trades_history[-1]
            last_trade['pnl'] = position.pl
            last_trade['exit_price'] = exit_price
            last_trade['exit_time'] = exit_time
        
        # Log trade result
        logger.info(f"Trade exited: P&L={position.pl:.4f}, "
                   f"Consecutive losses={self.consecutive_losses}")
        
        # Check if we should reduce position sizes after consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            logger.warning(f"Consecutive losses limit reached: {self.consecutive_losses}")
            # Could implement position size reduction logic here
