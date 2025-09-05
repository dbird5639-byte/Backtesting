"""
Market Data Manager

Handles real-time and historical market data from multiple sources
including Yahoo Finance, Alpaca, and other providers.
Provides caching, data validation, and technical indicator calculation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data"""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data"""
        pass


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        super().__init__("yahoo_finance")
    
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns for {symbol}: {missing_columns}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                return None
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=info.get('regularMarketOpen', 0),
                high=info.get('dayHigh', 0),
                low=info.get('dayLow', 0),
                close=info['regularMarketPrice'],
                volume=info.get('volume', 0),
                source=self.name
            )
            
        except Exception as e:
            logger.error(f"Error getting real-time Yahoo Finance data for {symbol}: {e}")
            return None


class CSVDataSource(DataSource):
    """Local CSV data source for the Advanced Backtesting system"""
    
    def __init__(self, data_dir: str = "Data"):
        super().__init__("csv_local")
        self.data_dir = data_dir
    
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data from local CSV files"""
        try:
            # Find matching CSV file
            pattern = f"{symbol}_{interval}_*candles_*.csv"
            import glob
            files = glob.glob(os.path.join(self.data_dir, pattern))
            
            if not files:
                logger.warning(f"No CSV files found for {symbol}_{interval}")
                return pd.DataFrame()
            
            # Use the most recent file
            latest_file = max(files, key=os.path.getctime)
            
            # Load data
            data = pd.read_csv(latest_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Rename columns to standard format
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            data.rename(columns=column_mapping, inplace=True)
            
            # Filter by date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            logger.info(f"Loaded {len(data)} data points for {symbol} from {latest_file}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data from CSV files (not real-time)"""
        try:
            # Get the most recent data point
            data = self.get_historical_data(symbol, "", "", "1h")
            
            if data.empty:
                return None
            
            latest = data.iloc[-1]
            
            return MarketData(
                symbol=symbol,
                timestamp=data.index[-1],
                open=latest['open'],
                high=latest['high'],
                low=latest['low'],
                close=latest['close'],
                volume=latest['volume'],
                source=self.name
            )
            
        except Exception as e:
            logger.error(f"Error getting latest CSV data for {symbol}: {e}")
            return None


class MarketDataManager:
    """
    Main market data manager for the Advanced Backtesting system.
    
    Provides unified interface for multiple data sources with caching,
    data validation, and technical indicator calculation.
    """
    
    def __init__(self, db_path: str = "data/market_data.db", cache_dir: str = "data/cache"):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.data_sources = {}
        self.subscriptions = {}
        self.realtime_data = {}
        self.data_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Create directories
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Add default data sources
        self.add_data_source(CSVDataSource())
        self.add_data_source(YahooFinanceSource())
        
        logger.info("Market Data Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for data storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def add_data_source(self, source: DataSource):
        """Add a data source to the manager."""
        self.data_sources[source.name] = source
        logger.info(f"Added data source: {source.name}")
    
    def get_historical_data(self, symbol: str, start_date: str = "", 
                           end_date: str = "", interval: str = "1d",
                           source: str = "csv_local", use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical market data from specified source.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            source: Data source name
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            if use_cache:
                cache_key = f"{symbol}_{start_date}_{end_date}_{interval}_{source}"
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info(f"Retrieved cached data for {symbol}")
                    return cached_data
            
            # Get data from source
            if source not in self.data_sources:
                logger.error(f"Data source not found: {source}")
                return pd.DataFrame()
            
            data_source = self.data_sources[source]
            data = data_source.get_historical_data(symbol, start_date, end_date, interval)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol} from {source}")
                return data
            
            # Validate data
            if not self._validate_data(data):
                logger.error(f"Data validation failed for {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            if use_cache:
                self._save_to_cache(cache_key, data)
            
            # Save to database
            self._save_to_database(symbol, data, source)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate market data quality."""
        try:
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if (data[col] < 0).any():
                    logger.error(f"Negative prices found in {col}")
                    return False
            
            # Check OHLC relationships
            invalid_high = data['high'] < data[['open', 'close']].max(axis=1)
            invalid_low = data['low'] > data[['open', 'close']].min(axis=1)
            
            if invalid_high.any() or invalid_low.any():
                logger.error("Invalid OHLC relationships detected")
                return False
            
            # Check for excessive missing data
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_pct > 0.1:  # More than 10% missing
                logger.warning(f"High percentage of missing data: {missing_pct:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        try:
            # Check memory cache first
            if cache_key in self.data_cache:
                cache_entry = self.data_cache[cache_key]
                if datetime.now() < cache_entry['expires_at']:
                    return cache_entry['data']
                else:
                    del self.data_cache[cache_key]
            
            # Check database cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT data, expires_at FROM data_cache 
                WHERE cache_key = ? AND expires_at > ?
            ''', (cache_key, datetime.now().isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data_json, expires_at = result
                data = pd.read_json(data_json)
                
                # Add to memory cache
                self.data_cache[cache_key] = {
                    'data': data,
                    'expires_at': datetime.fromisoformat(expires_at)
                }
                
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        try:
            # Save to memory cache
            expires_at = datetime.now() + timedelta(seconds=self.cache_ttl)
            self.data_cache[cache_key] = {
                'data': data,
                'expires_at': expires_at
            }
            
            # Save to database cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data_json = data.to_json()
            
            cursor.execute('''
                INSERT OR REPLACE INTO data_cache (cache_key, data, expires_at)
                VALUES (?, ?, ?)
            ''', (cache_key, data_json, expires_at.isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _save_to_database(self, symbol: str, data: pd.DataFrame, source: str):
        """Save data to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for timestamp, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timestamp.isoformat(),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    source
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def get_data_from_database(self, symbol: str, start_date: str, 
                              end_date: str) -> pd.DataFrame:
        """Get data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume, source
                FROM market_data 
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            data = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            if not data.empty:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting data from database: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for market data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            if data.empty:
                return data
            
            # Make a copy to avoid modifying original
            df = data.copy()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price-based indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Support and resistance
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            logger.info(f"Calculated technical indicators for {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market summary for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with market summary
        """
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "symbols": {},
                "market_overview": {}
            }
            
            for symbol in symbols:
                # Get latest data
                data = self.get_historical_data(symbol, "", "", "1d", "csv_local")
                
                if not data.empty:
                    latest = data.iloc[-1]
                    
                    # Calculate basic metrics
                    returns = data['close'].pct_change().dropna()
                    
                    symbol_summary = {
                        "current_price": latest['close'],
                        "change": (latest['close'] / data['close'].iloc[-2] - 1) * 100 if len(data) > 1 else 0,
                        "volume": latest['volume'],
                        "high_52w": data['high'].max(),
                        "low_52w": data['low'].min(),
                        "volatility": returns.std() * np.sqrt(252) * 100,
                        "data_points": len(data)
                    }
                    
                    summary["symbols"][symbol] = symbol_summary
            
            # Market overview
            if summary["symbols"]:
                prices = [s["current_price"] for s in summary["symbols"].values()]
                changes = [s["change"] for s in summary["symbols"].values()]
                
                summary["market_overview"] = {
                    "avg_price": np.mean(prices),
                    "avg_change": np.mean(changes),
                    "advancing": len([c for c in changes if c > 0]),
                    "declining": len([c for c in changes if c < 0]),
                    "unchanged": len([c for c in changes if c == 0])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries."""
        try:
            # Clean memory cache
            current_time = datetime.now()
            expired_keys = [
                key for key, entry in self.data_cache.items()
                if current_time > entry['expires_at']
            ]
            
            for key in expired_keys:
                del self.data_cache[key]
            
            # Clean database cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (current_time - timedelta(hours=max_age_hours)).isoformat()
            cursor.execute('DELETE FROM data_cache WHERE expires_at < ?', (cutoff_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up cache: {len(expired_keys)} memory entries, {deleted_count} database entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from CSV data source."""
        try:
            if "csv_local" not in self.data_sources:
                return []
            
            csv_source = self.data_sources["csv_local"]
            data_dir = csv_source.data_dir
            
            if not os.path.exists(data_dir):
                return []
            
            symbols = set()
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv'):
                    parts = filename.split('_')
                    if len(parts) >= 1:
                        symbols.add(parts[0])
            
            return sorted(list(symbols))
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_symbol_timeframes(self, symbol: str) -> List[str]:
        """Get available timeframes for a symbol."""
        try:
            if "csv_local" not in self.data_sources:
                return []
            
            csv_source = self.data_sources["csv_local"]
            data_dir = csv_source.data_dir
            
            if not os.path.exists(data_dir):
                return []
            
            timeframes = set()
            for filename in os.listdir(data_dir):
                if filename.startswith(f"{symbol}_") and filename.endswith('.csv'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        timeframes.add(parts[1])
            
            return sorted(list(timeframes))
            
        except Exception as e:
            logger.error(f"Error getting timeframes for {symbol}: {e}")
            return []


def main():
    """Example usage of the Market Data Manager."""
    # Initialize manager
    manager = MarketDataManager()
    
    # Get available symbols
    symbols = manager.get_available_symbols()
    print(f"Available symbols: {symbols}")
    
    # Get data for a symbol
    if symbols:
        symbol = symbols[0]
        data = manager.get_historical_data(symbol, "", "", "1h")
        
        if not data.empty:
            print(f"\nData for {symbol}:")
            print(f"  Rows: {len(data)}")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            print(f"  Columns: {list(data.columns)}")
            
            # Calculate technical indicators
            data_with_indicators = manager.calculate_technical_indicators(data)
            print(f"  Technical indicators added: {len(data_with_indicators.columns)} columns")
            
            # Get market summary
            summary = manager.get_market_summary([symbol])
            print(f"\nMarket summary: {summary}")
    
    # Cleanup
    manager.cleanup_cache()


if __name__ == "__main__":
    main() 