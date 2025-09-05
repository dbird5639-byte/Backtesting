#!/usr/bin/env python3
"""
Crypto Calendar Bull Run Strategy

A sophisticated strategy that combines:
- Calendar-based event trading (halvings, ETF approvals, institutional adoption)
- Bull run momentum capture
- Multi-timeframe analysis
- Dynamic position sizing based on market phases
- Risk management optimized for bull market conditions
- Integration with existing test engines

This strategy is specifically designed for the upcoming crypto bull run period.
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List, Optional, Tuple
from backtesting import Strategy
from enum import Enum
from datetime import datetime, timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BullRunPhase(Enum):
    """Bull run phases for strategic positioning"""
    ACCUMULATION = "accumulation"      # Early phase - accumulate positions
    MARKUP = "markup"                  # Main bull run - trend following
    DISTRIBUTION = "distribution"      # Late phase - take profits
    TRANSITION = "transition"          # Phase changes

class CalendarEvent(Enum):
    """Key calendar events that drive crypto markets"""
    HALVING = "halving"                # Bitcoin halving events
    ETF_APPROVAL = "etf_approval"      # ETF approval dates
    INSTITUTIONAL = "institutional"    # Institutional adoption milestones
    REGULATORY = "regulatory"          # Regulatory decision dates
    TECHNICAL = "technical"            # Technical analysis levels

class CryptoCalendarBullRunStrategy(Strategy):
    """
    Crypto Calendar Bull Run Strategy
    
    Combines calendar events with technical analysis for optimal bull run positioning:
    - Calendar-driven entry/exit timing
    - Bull run phase detection and adaptation
    - Multi-timeframe momentum confirmation
    - Dynamic position sizing based on market phases
    - Advanced risk management for bull market conditions
    """
    
    # Core parameters
    risk_per_trade = 0.01              # 1% risk per trade (bull market aggressive)
    max_positions = 5                  # Maximum concurrent positions (bull market)
    max_drawdown = 0.12                # Maximum drawdown limit (12% for bull market)
    consecutive_loss_limit = 4         # Maximum consecutive losses
    
    # Bull run phase detection
    phase_lookback = 100               # Periods to analyze for phase detection
    phase_threshold = 0.7              # Confidence threshold for phase
    phase_smoothing = 10               # Smoothing period for phase signals
    
    # Multi-timeframe analysis
    timeframes = [20, 50, 200]        # EMA periods for different timeframes
    momentum_periods = [14, 21, 50]   # RSI periods for different timeframes
    
    # Technical indicators
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
    # Volatility and momentum
    atr_period = 14
    atr_multiplier = 2.5              # Wider stops for bull market
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Volume analysis
    volume_period = 20
    volume_threshold = 1.8             # Higher volume threshold for bull market
    
    # Calendar event parameters
    event_lookforward = 30             # Days to look forward for events
    event_impact_duration = 7          # Days of impact for each event
    pre_event_position_multiplier = 1.5 # Increase position size before events
    
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
        self.current_phase = BullRunPhase.ACCUMULATION
        self.phase_confidence = 0.0
        
        # Calendar events tracking
        self.calendar_events = self._initialize_calendar_events()
        self.event_positions = {}      # Track positions related to events
        
        # Multi-timeframe indicators
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=self.timeframes[0])
        self.ema_50 = self.I(talib.EMA, self.data.Close, timeperiod=self.timeframes[1])
        self.ema_200 = self.I(talib.EMA, self.data.Close, timeperiod=self.timeframes[2])
        
        # Momentum indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.rsi_21 = self.I(talib.RSI, self.data.Close, timeperiod=self.momentum_periods[1])
        self.rsi_50 = self.I(talib.RSI, self.data.Close, timeperiod=self.momentum_periods[2])
        
        # MACD for trend confirmation
        self.macd, self.macd_signal_line, self.macd_histogram = self.I(
            talib.MACD, self.data.Close, 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        # Volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        
        # Volume
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
        # Bollinger Bands for volatility analysis
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Stochastic for momentum
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, self.data.High, self.data.Low, self.data.Close,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
    
    def _initialize_calendar_events(self) -> Dict:
        """Initialize key calendar events for crypto markets"""
        current_year = datetime.now().year
        
        events = {
            CalendarEvent.HALVING: {
                'dates': [
                    datetime(current_year, 4, 20),  # Bitcoin halving (approximate)
                    datetime(current_year + 4, 4, 20),  # Next halving
                ],
                'impact': 'high',
                'description': 'Bitcoin halving event'
            },
            CalendarEvent.ETF_APPROVAL: {
                'dates': [
                    datetime(current_year, 1, 10),  # Spot ETF approval anniversary
                    datetime(current_year, 3, 15),  # Potential new ETF approvals
                    datetime(current_year, 6, 15),  # Mid-year ETF developments
                    datetime(current_year, 9, 15),  # Q3 ETF updates
                ],
                'impact': 'high',
                'description': 'ETF approval and development dates'
            },
            CalendarEvent.INSTITUTIONAL: {
                'dates': [
                    datetime(current_year, 1, 1),   # New year institutional flows
                    datetime(current_year, 4, 1),   # Q2 institutional positioning
                    datetime(current_year, 7, 1),   # Q3 institutional flows
                    datetime(current_year, 10, 1),  # Q4 institutional positioning
                ],
                'impact': 'medium',
                'description': 'Institutional adoption milestones'
            },
            CalendarEvent.REGULATORY: {
                'dates': [
                    datetime(current_year, 3, 1),   # Q1 regulatory decisions
                    datetime(current_year, 6, 1),   # Mid-year regulatory updates
                    datetime(current_year, 9, 1),   # Q3 regulatory decisions
                    datetime(current_year, 12, 1),  # Year-end regulatory clarity
                ],
                'impact': 'medium',
                'description': 'Regulatory decision dates'
            },
            CalendarEvent.TECHNICAL: {
                'dates': [
                    datetime(current_year, 1, 1),   # Yearly technical levels
                    datetime(current_year, 4, 1),   # Q2 technical analysis
                    datetime(current_year, 7, 1),   # Q3 technical levels
                    datetime(current_year, 10, 1),  # Q4 technical analysis
                ],
                'impact': 'low',
                'description': 'Technical analysis review dates'
            }
        }
        
        return events
    
    def detect_bull_run_phase(self) -> Tuple[BullRunPhase, float]:
        """Detect the current phase of the bull run"""
        if len(self.data) < self.phase_lookback:
            return BullRunPhase.ACCUMULATION, 0.5
        
        # Price relative to EMAs
        price_above_ema20 = self.data.Close[-1] > self.ema_20[-1]
        price_above_ema50 = self.data.Close[-1] > self.ema_50[-1]
        price_above_ema200 = self.data.Close[-1] > self.ema_200[-1]
        
        # EMA alignment
        ema20_above_ema50 = self.ema_20[-1] > self.ema_50[-1]
        ema50_above_ema200 = self.ema_50[-1] > self.ema_200[-1]
        
        # Momentum indicators
        rsi_strong = self.rsi[-1] > 50
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        
        # Volume confirmation
        volume_strong = self.data.Volume[-1] > self.volume_ma[-1] * 1.5
        
        # Calculate phase confidence
        confidence = 0.0
        
        if price_above_ema20 and price_above_ema50 and price_above_ema200:
            confidence += 0.3
        if ema20_above_ema50 and ema50_above_ema200:
            confidence += 0.3
        if rsi_strong and macd_positive:
            confidence += 0.2
        if volume_strong:
            confidence += 0.2
        
        # Determine phase based on confidence and market conditions
        if confidence >= 0.8:
            if self.rsi[-1] > 70:
                return BullRunPhase.DISTRIBUTION, confidence
            else:
                return BullRunPhase.MARKUP, confidence
        elif confidence >= 0.6:
            return BullRunPhase.ACCUMULATION, confidence
        else:
            return BullRunPhase.TRANSITION, confidence
    
    def get_phase_specific_parameters(self) -> Dict:
        """Get strategy parameters specific to the current bull run phase"""
        base_params = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 2.0,
            'entry_threshold': 0.7,
            'max_hold_time': 48,
            'risk_multiplier': 1.0
        }
        
        if self.current_phase == BullRunPhase.ACCUMULATION:
            base_params.update({
                'position_size_multiplier': 0.8,      # Moderate accumulation
                'stop_loss_multiplier': 0.8,          # Tighter stops
                'take_profit_multiplier': 1.5,        # Lower profit targets
                'entry_threshold': 0.8,               # Very selective entries
                'max_hold_time': 72,                  # Longer holds for accumulation
                'risk_multiplier': 0.8                # Lower risk
            })
        elif self.current_phase == BullRunPhase.MARKUP:
            base_params.update({
                'position_size_multiplier': 1.5,      # Aggressive positioning
                'stop_loss_multiplier': 1.2,          # Wider stops for volatility
                'take_profit_multiplier': 3.0,        # Higher profit targets
                'entry_threshold': 0.6,               # More entry opportunities
                'max_hold_time': 36,                  # Moderate hold time
                'risk_multiplier': 1.2                # Higher risk tolerance
            })
        elif self.current_phase == BullRunPhase.DISTRIBUTION:
            base_params.update({
                'position_size_multiplier': 0.6,      # Reduced position sizes
                'stop_loss_multiplier': 0.6,          # Very tight stops
                'take_profit_multiplier': 1.2,        # Quick profits
                'entry_threshold': 0.9,               # Very selective entries
                'max_hold_time': 24,                  # Short holds
                'risk_multiplier': 0.6                # Lower risk
            })
        else:  # TRANSITION
            base_params.update({
                'position_size_multiplier': 0.4,      # Minimal positions
                'stop_loss_multiplier': 0.5,          # Very tight stops
                'take_profit_multiplier': 1.1,        # Quick profits
                'entry_threshold': 0.95,              # Extremely selective
                'max_hold_time': 12,                  # Very short holds
                'risk_multiplier': 0.5                # Minimal risk
            })
        
        return base_params
    
    def check_calendar_events(self) -> Dict:
        """Check for upcoming calendar events and their impact"""
        current_date = datetime.now()
        upcoming_events = {}
        
        for event_type, event_data in self.calendar_events.items():
            for event_date in event_data['dates']:
                days_until = (event_date - current_date).days
                
                if 0 <= days_until <= self.event_lookforward:
                    impact_score = self._calculate_event_impact(event_type, days_until)
                    upcoming_events[event_type] = {
                        'date': event_date,
                        'days_until': days_until,
                        'impact_score': impact_score,
                        'description': event_data['description']
                    }
        
        return upcoming_events
    
    def _calculate_event_impact(self, event_type: CalendarEvent, days_until: int) -> float:
        """Calculate the impact score of a calendar event"""
        base_impact = {
            CalendarEvent.HALVING: 0.9,
            CalendarEvent.ETF_APPROVAL: 0.8,
            CalendarEvent.INSTITUTIONAL: 0.6,
            CalendarEvent.REGULATORY: 0.5,
            CalendarEvent.TECHNICAL: 0.3
        }
        
        # Impact decreases as event gets further away
        time_decay = max(0.1, 1.0 - (days_until / self.event_lookforward))
        
        # Pre-event positioning bonus
        if days_until <= 3:
            time_decay *= 1.5
        
        return base_impact[event_type] * time_decay
    
    def should_enter_long(self) -> bool:
        """Determine if we should enter a long position"""
        if len(self.data) < max(self.timeframes[2], self.rsi_period):
            return False
        
        params = self.get_phase_specific_parameters()
        calendar_events = self.check_calendar_events()
        
        # Base technical conditions
        price_above_ema20 = self.data.Close[-1] > self.ema_20[-1]
        ema_trending_up = self.ema_20[-1] > self.ema_20[-2]
        rsi_not_overbought = self.rsi[-1] < self.rsi_overbought
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_threshold
        
        # Multi-timeframe momentum
        momentum_aligned = (self.rsi_21[-1] > 45 and self.rsi_50[-1] > 40)
        
        # MACD confirmation
        macd_positive = self.macd[-1] > self.macd_signal_line[-1]
        macd_rising = self.macd[-1] > self.macd[-2]
        
        # Calendar event boost
        calendar_boost = 0.0
        if calendar_events:
            max_impact = max([event['impact_score'] for event in calendar_events.values()])
            calendar_boost = max_impact * 0.3
        
        # Calculate entry probability
        entry_probability = 0.0
        
        if price_above_ema20 and ema_trending_up:
            entry_probability += 0.3
        if rsi_not_overbought and momentum_aligned:
            entry_probability += 0.3
        if volume_above_average:
            entry_probability += 0.2
        if macd_positive and macd_rising:
            entry_probability += 0.2
        
        # Add calendar boost
        entry_probability += calendar_boost
        
        return entry_probability >= params['entry_threshold']
    
    def should_enter_short(self) -> bool:
        """Determine if we should enter a short position (limited in bull markets)"""
        if len(self.data) < max(self.timeframes[2], self.rsi_period):
            return False
        
        # In bull markets, only take shorts during distribution or transition phases
        if self.current_phase not in [BullRunPhase.DISTRIBUTION, BullRunPhase.TRANSITION]:
            return False
        
        params = self.get_phase_specific_parameters()
        
        # Very selective short conditions
        price_below_ema20 = self.data.Close[-1] < self.ema_20[-1]
        ema_trending_down = self.ema_20[-1] < self.ema_20[-2]
        rsi_overbought = self.rsi[-1] > 80
        volume_above_average = self.data.Volume[-1] > self.volume_ma[-1] * 2.0
        
        # MACD bearish
        macd_negative = self.macd[-1] < self.macd_signal_line[-1]
        macd_falling = self.macd[-1] < self.macd[-2]
        
        # Stochastic overbought
        stoch_overbought = self.stoch_k[-1] > 80 and self.stoch_d[-1] > 80
        
        return (price_below_ema20 and ema_trending_down and rsi_overbought and 
                volume_above_average and macd_negative and macd_falling and stoch_overbought)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion and phase adjustment"""
        if stop_loss == entry_price:
            return 0.0
        
        params = self.get_phase_specific_parameters()
        calendar_events = self.check_calendar_events()
        
        # Calculate base position size
        risk_amount = self.equity * self.risk_per_trade * params['risk_multiplier']
        price_risk = abs(entry_price - stop_loss)
        base_position_size = risk_amount / price_risk
        
        # Apply phase-specific multiplier
        adjusted_position_size = base_position_size * params['position_size_multiplier']
        
        # Apply calendar event multiplier
        if calendar_events:
            max_impact = max([event['impact_score'] for event in calendar_events.values()])
            if max_impact > 0.7:  # High impact events
                adjusted_position_size *= self.pre_event_position_multiplier
        
        # Ensure we don't exceed max positions
        current_positions = len(self.positions)
        if current_positions >= self.max_positions:
            adjusted_position_size *= 0.5
        
        return adjusted_position_size
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss based on phase and volatility"""
        params = self.get_phase_specific_parameters()
        atr_value = self.atr[-1]
        
        if is_long:
            stop_loss = entry_price - (atr_value * self.atr_multiplier * params['stop_loss_multiplier'])
        else:
            stop_loss = entry_price + (atr_value * self.atr_multiplier * params['stop_loss_multiplier'])
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, is_long: bool) -> float:
        """Calculate take profit based on phase and risk-reward"""
        params = self.get_phase_specific_parameters()
        risk = abs(entry_price - stop_loss)
        
        if is_long:
            take_profit = entry_price + (risk * params['take_profit_multiplier'])
        else:
            take_profit = entry_price - (risk * params['take_profit_multiplier'])
        
        return take_profit
    
    def next(self):
        """Main strategy logic executed on each bar"""
        # Update bull run phase
        self.current_phase, self.phase_confidence = self.detect_bull_run_phase()
        
        # Log phase changes
        if len(self.trades_history) == 0 or self.trades_history[-1].get('phase') != self.current_phase.value:
            logger.info(f"Bull run phase changed to: {self.current_phase.value} (confidence: {self.phase_confidence:.2f})")
        
        # Check calendar events
        calendar_events = self.check_calendar_events()
        if calendar_events:
            events_str = [f"{k.value}: {v['days_until']} days" for k, v in calendar_events.items()]
            logger.info(f"Upcoming calendar events: {events_str}")
        
        # Check for exit conditions on existing positions
        self.manage_existing_positions()
        
        # Check for new entry opportunities
        if len(self.positions) < self.max_positions:
            self.check_entry_opportunities()
    
    def check_entry_opportunities(self):
        """Check for new entry opportunities based on current phase and calendar"""
        current_price = self.data.Close[-1]
        calendar_events = self.check_calendar_events()
        
        # Long entry
        if self.should_enter_long():
            stop_loss = self.calculate_stop_loss(current_price, True)
            take_profit = self.calculate_take_profit(current_price, stop_loss, True)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("LONG", current_price, stop_loss, take_profit, position_size, calendar_events)
        
        # Short entry (limited in bull markets)
        elif self.should_enter_short():
            stop_loss = self.calculate_stop_loss(current_price, False)
            take_profit = self.calculate_take_profit(current_price, stop_loss, False)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                self.record_trade("SHORT", current_price, stop_loss, take_profit, position_size, calendar_events)
    
    def manage_existing_positions(self):
        """Manage existing positions with phase-specific logic"""
        for position in self.positions:
            # Check if position has been held too long
            params = self.get_phase_specific_parameters()
            
            # Phase-specific position management
            if self.current_phase == BullRunPhase.DISTRIBUTION and position.pl > 0.02:
                # Take profits more aggressively in distribution phase
                self.position.close()
                logger.info(f"Closing profitable position early due to distribution phase")
            
            elif self.current_phase == BullRunPhase.TRANSITION and position.pl < -0.01:
                # Cut losses quickly in transition phase
                self.position.close()
                logger.info(f"Closing losing position early due to transition phase")
    
    def record_trade(self, direction: str, entry_price: float, stop_loss: float, 
                    take_profit: float, position_size: float, calendar_events: Dict):
        """Record trade details for analysis"""
        trade_info = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'phase': self.current_phase.value,
            'phase_confidence': self.phase_confidence,
            'calendar_events': [k.value for k in calendar_events.keys()] if calendar_events else [],
            'timestamp': len(self.data),
            'equity': self.equity
        }
        self.trades_history.append(trade_info)
        
        logger.info(f"Entered {direction} position: Entry={entry_price:.4f}, "
                   f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                   f"Phase={self.current_phase.value}, "
                   f"Calendar events={len(calendar_events)}")
    
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
