"""
Abstract Base Data Handler for Backtesting

This module provides the abstract base class for all data handlers.
All data handlers must implement the methods defined in this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging


@dataclass
class DataConfig:
    """Configuration for data handling"""
    
    # Data source settings
    data_source: str = "local"  # local, api, database
    data_format: str = "csv"     # csv, parquet, json, database
    
    # Data paths
    base_data_path: str = "./data"
    cache_path: str = "./cache"
    
    # Data quality settings
    min_data_points: int = 100
    max_data_points: int = 100000
    required_columns: List[str] = None
    
    # Data preprocessing
    fill_missing: bool = True
    remove_outliers: bool = False
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Caching
    enable_caching: bool = True
    cache_expiry_hours: int = 24
    
    # Validation
    validate_ohlcv: bool = True
    check_data_integrity: bool = True
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']


@dataclass
class DataInfo:
    """Information about loaded data"""
    
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_rows: int
    total_columns: int
    missing_values: int
    data_quality_score: float  # 0.0 to 1.0
    last_updated: datetime
    source: str
    file_size_mb: float = 0.0


class BaseDataHandler(ABC):
    """
    Abstract base class for all data handlers.
    
    This class defines the interface that all data handlers must implement.
    It provides common functionality for data loading, validation, and management.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the base data handler.
        
        Args:
            config: Data handler configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the data handler"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        Path(self.config.base_data_path).mkdir(parents=True, exist_ok=True)
        if self.config.enable_caching:
            Path(self.config.cache_path).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate data handler configuration"""
        if not self.config.base_data_path:
            raise ValueError("Base data path is required")
        
        if self.config.min_data_points <= 0:
            raise ValueError("Minimum data points must be positive")
        
        if self.config.max_data_points <= self.config.min_data_points:
            raise ValueError("Maximum data points must be greater than minimum")
    
    @abstractmethod
    def load_data(self, symbol: str, timeframe: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Load market data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            timeframe: Data timeframe (e.g., '1h', '1d')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            force_reload: Force reload from source (ignore cache)
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, symbol: str, timeframe: str, 
                  format: Optional[str] = None) -> bool:
        """
        Save market data to storage.
        
        Args:
            data: DataFrame to save
            symbol: Trading symbol
            timeframe: Data timeframe
            format: Output format (optional, uses config default if not specified)
            
        Returns:
            True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_available_data(self, symbol: Optional[str] = None, 
                           timeframe: Optional[str] = None) -> List[DataInfo]:
        """
        List available data files/datasets.
        
        Args:
            symbol: Filter by symbol (optional)
            timeframe: Filter by timeframe (optional)
            
        Returns:
            List of data information objects
        """
        pass
    
    def get_data_info(self, symbol: str, timeframe: str) -> Optional[DataInfo]:
        """
        Get information about a specific dataset.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            DataInfo object if found, None otherwise
        """
        available_data = self.list_available_data(symbol, timeframe)
        if available_data:
            return available_data[0]
        return None
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality and integrity.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        missing_columns = set(self.config.required_columns) - set(data.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                issues.append(f"Column {col} is not numeric")
        
        # Check for missing values
        missing_count = data[self.config.required_columns].isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values")
        
        # Check data integrity
        if self.config.validate_ohlcv:
            ohlc_issues = self._validate_ohlcv(data)
            issues.extend(ohlc_issues)
        
        # Check data length
        if len(data) < self.config.min_data_points:
            issues.append(f"Data has {len(data)} rows, minimum required is {self.config.min_data_points}")
        
        if len(data) > self.config.max_data_points:
            issues.append(f"Data has {len(data)} rows, maximum allowed is {self.config.max_data_points}")
        
        return len(issues) == 0, issues
    
    def _validate_ohlcv(self, data: pd.DataFrame) -> List[str]:
        """Validate OHLCV data integrity"""
        issues = []
        
        if 'High' in data.columns and 'Low' in data.columns:
            # High should always be >= Low
            invalid_hl = data[data['High'] < data['Low']]
            if len(invalid_hl) > 0:
                issues.append(f"Found {len(invalid_hl)} rows where High < Low")
        
        if 'Open' in data.columns and 'Close' in data.columns and 'High' in data.columns and 'Low' in data.columns:
            # High should be >= max(Open, Close)
            # Low should be <= min(Open, Close)
            invalid_high = data[data['High'] < data[['Open', 'Close']].max(axis=1)]
            invalid_low = data[data['Low'] > data[['Open', 'Close']].min(axis=1)]
            
            if len(invalid_high) > 0:
                issues.append(f"Found {len(invalid_high)} rows where High < max(Open, Close)")
            if len(invalid_low) > 0:
                issues.append(f"Found {len(invalid_low)} rows where Low > min(Open, Close)")
        
        return issues
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for backtesting.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        processed_data = data.copy()
        
        # Fill missing values
        if self.config.fill_missing:
            processed_data = self._fill_missing_values(processed_data)
        
        # Remove outliers
        if self.config.remove_outliers:
            processed_data = self._remove_outliers(processed_data)
        
        # Ensure proper data types
        processed_data = self._ensure_data_types(processed_data)
        
        # Sort by timestamp
        if 'timestamp' in processed_data.columns:
            processed_data = processed_data.sort_values('timestamp')
        elif processed_data.index.name == 'timestamp':
            processed_data = processed_data.sort_index()
        
        return processed_data
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the data"""
        # Forward fill for OHLCV data
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # Backward fill for any remaining missing values
        data = data.fillna(method='bfill')
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data"""
        # Remove outliers based on price changes
        if 'Close' in data.columns:
            price_changes = data['Close'].pct_change().abs()
            outlier_threshold = price_changes.mean() + self.config.outlier_threshold * price_changes.std()
            outlier_mask = price_changes > outlier_threshold
            
            if outlier_mask.sum() > 0:
                self.logger.warning(f"Removing {outlier_mask.sum()} outlier rows")
                data = data[~outlier_mask]
        
        return data
    
    def _ensure_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for all columns"""
        # Convert OHLCV to float
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert timestamp to datetime if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        
        return data
    
    def get_cache_key(self, symbol: str, timeframe: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> str:
        """Generate cache key for data"""
        key_parts = [symbol, timeframe]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        return "_".join(key_parts)
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if not self.config.enable_caching:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age < timedelta(hours=self.config.cache_expiry_hours)
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache entries"""
        if cache_key is None:
            self.cache.clear()
            self.cache_timestamps.clear()
            self.logger.info("Cleared all cache entries")
        else:
            if cache_key in self.cache:
                del self.cache[cache_key]
            if cache_key in self.cache_timestamps:
                del self.cache_timestamps[cache_key]
            self.logger.info(f"Cleared cache for key: {cache_key}")
    
    def get_data_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get summary statistics for a dataset.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            data = self.load_data(symbol, timeframe)
            if data.empty:
                return {}
            
            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'total_rows': len(data),
                'date_range': {
                    'start': data.index.min() if hasattr(data.index, 'min') else None,
                    'end': data.index.max() if hasattr(data.index, 'max') else None
                },
                'columns': list(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
            
            # Add price statistics if available
            if 'Close' in data.columns:
                summary['price_stats'] = {
                    'min': float(data['Close'].min()),
                    'max': float(data['Close'].max()),
                    'mean': float(data['Close'].mean()),
                    'std': float(data['Close'].std())
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}
    
    def cleanup(self):
        """Clean up resources and clear cache"""
        self.logger.info("Cleaning up data handler resources")
        self.clear_cache()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
