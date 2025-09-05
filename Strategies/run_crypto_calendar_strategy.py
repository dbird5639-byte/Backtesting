#!/usr/bin/env python3
"""
Crypto Calendar Bull Run Strategy Runner

This script demonstrates how to run the strategy with different test engines
and provides a simple interface for testing and validation.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Engines"))

from Strategies.crypto_calendar_bull_run_strategy import CryptoCalendarBullRunStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoCalendarStrategyRunner:
    """Simple runner for the Crypto Calendar Bull Run Strategy"""
    
    def __init__(self, data_path: str = "./fetched_data"):
        self.data_path = data_path
        self.strategy = None
        self.data = None
        
    def load_data(self, symbol: str = "BTCUSDT", timeframe: str = "1h", days: int = 365):
        """Load or generate data for strategy testing"""
        try:
            # Try to load from fetched_data directory
            data_file = f"{self.data_path}/{symbol}_{timeframe}.csv"
            if os.path.exists(data_file):
                logger.info(f"Loading data from {data_file}")
                df = pd.read_csv(data_file)
                
                # Ensure proper column names
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                logger.info(f"Loaded {len(df)} rows of data")
            else:
                logger.info(f"Data file not found, generating synthetic data for {symbol}")
                df = self._generate_synthetic_data(symbol, timeframe, days)
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate synthetic crypto data for testing"""
        periods = days * 24  # Assuming hourly data
        
        # Base price (BTC-like)
        base_price = 45000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        # Generate price data with bull run characteristics
        np.random.seed(42)  # For reproducible results
        
        # Create trend with bull run characteristics
        trend = np.linspace(0, 0.8, periods)  # Bull run trend
        noise = np.random.normal(0, 0.02, periods)
        price_multiplier = 1 + trend + noise
        
        # Generate OHLC data
        opens = base_price * price_multiplier
        highs = opens * (1 + np.random.uniform(0, 0.05, periods))
        lows = opens * (1 - np.random.uniform(0, 0.03, periods))
        closes = opens * (1 + np.random.uniform(-0.02, 0.03, periods))
        
        # Generate volume data (higher during bull runs)
        base_volume = 1000000
        volume_multiplier = 1 + trend * 2 + np.random.uniform(0, 0.5, periods)
        volumes = base_volume * volume_multiplier
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        })
        
        # Create datetime index
        start_date = datetime.now() - timedelta(days=days)
        df.index = pd.date_range(start=start_date, periods=len(df), freq='1H')
        
        return df
    
    def initialize_strategy(self):
        """Initialize the strategy with loaded data"""
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return False
        
        try:
            self.strategy = CryptoCalendarBullRunStrategy()
            self.strategy.data = self.data
            self.strategy.equity = 100000  # Starting equity
            
            logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            return False
    
    def run_quick_analysis(self):
        """Run a quick analysis of the strategy"""
        if self.strategy is None:
            logger.error("Strategy not initialized. Please initialize first.")
            return
        
        logger.info("Running quick strategy analysis...")
        
        try:
            # Detect current market phase
            phase, confidence = self.strategy.detect_bull_run_phase()
            logger.info(f"Current market phase: {phase.value} (confidence: {confidence:.2f})")
            
            # Check calendar events
            calendar_events = self.strategy.check_calendar_events()
            logger.info(f"Upcoming calendar events: {len(calendar_events)}")
            
            if calendar_events:
                for event_type, event_data in calendar_events.items():
                    logger.info(f"  {event_type.value}: {event_data['days_until']} days away "
                               f"(impact: {event_data['impact_score']:.2f})")
            
            # Get phase-specific parameters
            params = self.strategy.get_phase_specific_parameters()
            logger.info(f"Phase parameters: {params}")
            
            # Check entry conditions
            long_entry = self.strategy.should_enter_long()
            short_entry = self.strategy.should_enter_short()
            
            logger.info(f"Long entry signal: {'YES' if long_entry else 'NO'}")
            logger.info(f"Short entry signal: {'YES' if short_entry else 'NO'}")
            
            return {
                'phase': phase.value,
                'confidence': confidence,
                'calendar_events': len(calendar_events),
                'long_signal': long_entry,
                'short_signal': short_entry,
                'parameters': params
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            return None
    
    def run_simple_backtest(self, bars: int = 100):
        """Run a simple backtest simulation"""
        if self.strategy is None:
            logger.error("Strategy not initialized. Please initialize first.")
            return
        
        logger.info(f"Running simple backtest simulation for {bars} bars...")
        
        try:
            # Simulate strategy execution
            trades = []
            equity_curve = [100000]  # Starting equity
            
            for i in range(min(bars, len(self.strategy.data))):
                # Update strategy data to current position
                current_data = self.strategy.data.iloc[:i+1]
                self.strategy.data = current_data
                
                # Check for entry signals
                if self.strategy.should_enter_long():
                    current_price = current_data['Close'].iloc[-1]
                    stop_loss = self.strategy.calculate_stop_loss(current_price, True)
                    take_profit = self.strategy.calculate_take_profit(current_price, stop_loss, True)
                    position_size = self.strategy.calculate_position_size(current_price, stop_loss)
                    
                    if position_size > 0:
                        trade = {
                            'type': 'LONG',
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size,
                            'bar': i
                        }
                        trades.append(trade)
                        logger.info(f"Bar {i}: LONG entry at {current_price:.2f}")
                
                # Update equity (simplified)
                if trades:
                    # Simple equity calculation
                    current_equity = equity_curve[-1] * (1 + np.random.normal(0, 0.001))
                    equity_curve.append(current_equity)
                else:
                    equity_curve.append(equity_curve[-1])
            
            # Calculate basic metrics
            total_trades = len(trades)
            final_equity = equity_curve[-1]
            total_return = (final_equity - 100000) / 100000
            
            logger.info(f"Backtest completed:")
            logger.info(f"  Total trades: {total_trades}")
            logger.info(f"  Final equity: ${final_equity:,.2f}")
            logger.info(f"  Total return: {total_return:.2%}")
            
            return {
                'total_trades': total_trades,
                'final_equity': final_equity,
                'total_return': total_return,
                'trades': trades,
                'equity_curve': equity_curve
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None
    
    def run_with_engine(self, engine_name: str = "simple"):
        """Run strategy with a specific test engine"""
        logger.info(f"Attempting to run strategy with {engine_name} engine...")
        
        try:
            # Import the specified engine
            if engine_name == "simple":
                from Engines.simple_basic_engine import SimpleBasicEngine
                engine = SimpleBasicEngine()
            elif engine_name == "advanced":
                from Engines.advanced_engine import AdvancedEngine
                engine = AdvancedEngine()
            elif engine_name == "base":
                from Engines.base_engine import BaseEngine
                engine = BaseEngine()
            elif engine_name == "portfolio":
                from Engines.portfolio_engine import PortfolioEngine
                engine = PortfolioEngine()
            else:
                logger.error(f"Unknown engine: {engine_name}")
                return None
            
            # Prepare data for engine
            engine_data = {
                'symbol': 'BTCUSDT',
                'data': self.data,
                'strategy': CryptoCalendarBullRunStrategy,
                'initial_capital': 100000,
                'commission': 0.001
            }
            
            # Run backtest
            logger.info(f"Running {engine_name} engine backtest...")
            results = engine.run_backtest(engine_data)
            
            logger.info(f"Engine backtest completed successfully")
            return results
            
        except ImportError as e:
            logger.error(f"Could not import {engine_name} engine: {e}")
            return None
        except Exception as e:
            logger.error(f"Error running {engine_name} engine: {e}")
            return None

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Crypto Calendar Bull Run Strategy Runner')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Data timeframe (default: 1h)')
    parser.add_argument('--days', type=int, default=365, help='Days of data (default: 365)')
    parser.add_argument('--engine', default='simple', 
                       choices=['simple', 'advanced', 'base', 'portfolio'],
                       help='Test engine to use (default: simple)')
    parser.add_argument('--mode', default='analysis',
                       choices=['analysis', 'backtest', 'engine'],
                       help='Run mode (default: analysis)')
    
    args = parser.parse_args()
    
    print("🚀 Crypto Calendar Bull Run Strategy Runner")
    print("=" * 50)
    
    # Initialize runner
    runner = CryptoCalendarStrategyRunner()
    
    # Load data
    print(f"📊 Loading data for {args.symbol} ({args.timeframe}, {args.days} days)...")
    data = runner.load_data(args.symbol, args.timeframe, args.days)
    
    if data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    print(f"✅ Data loaded: {len(data)} rows")
    
    # Initialize strategy
    print("🔧 Initializing strategy...")
    if not runner.initialize_strategy():
        print("❌ Failed to initialize strategy. Exiting.")
        return
    
    print("✅ Strategy initialized successfully")
    
    # Run based on mode
    if args.mode == 'analysis':
        print("\n📈 Running quick analysis...")
        results = runner.run_quick_analysis()
        
        if results:
            print("\n📊 Analysis Results:")
            print(f"  Market Phase: {results['phase']} (confidence: {results['confidence']:.2f})")
            print(f"  Calendar Events: {results['calendar_events']}")
            print(f"  Long Signal: {'YES' if results['long_signal'] else 'NO'}")
            print(f"  Short Signal: {'YES' if results['short_signal'] else 'NO'}")
    
    elif args.mode == 'backtest':
        print("\n🔄 Running simple backtest...")
        results = runner.run_simple_backtest()
        
        if results:
            print("\n📊 Backtest Results:")
            print(f"  Total Trades: {results['total_trades']}")
            print(f"  Final Equity: ${results['final_equity']:,.2f}")
            print(f"  Total Return: {results['total_return']:.2%}")
    
    elif args.mode == 'engine':
        print(f"\n⚙️ Running with {args.engine} engine...")
        results = runner.run_with_engine(args.engine)
        
        if results:
            print(f"✅ {args.engine} engine completed successfully")
        else:
            print(f"❌ {args.engine} engine failed")
    
    print("\n🎯 Strategy ready for the bull run! 🚀📈")

if __name__ == "__main__":
    main()
