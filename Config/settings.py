"""
Main Application Settings

This module contains the main configuration settings for the backtesting system.
It provides a centralized way to manage all application settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

# Default configuration paths
DEFAULT_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "app_config.json"
DEFAULT_ENV_FILE = DEFAULT_CONFIG_DIR / ".env"


@dataclass
class DatabaseSettings:
    """Database connection settings"""
    
    # Database type
    database_type: str = "sqlite"  # sqlite, postgresql, mysql
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database_name: str = "backtesting"
    
    # SQLite specific
    sqlite_path: str = "./data/backtesting.db"
    
    # Connection pool
    max_connections: int = 10
    connection_timeout: int = 30
    
    # SSL settings
    use_ssl: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""


@dataclass
class APISettings:
    """API configuration settings"""
    
    # External APIs
    alpha_vantage_key: str = ""
    polygon_key: str = ""
    yahoo_finance_enabled: bool = True
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Timeout settings
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Cache settings
    api_cache_enabled: bool = True
    api_cache_duration: int = 3600  # 1 hour


@dataclass
class LoggingSettings:
    """Logging configuration settings"""
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log files
    log_dir: str = "./logs"
    log_file: str = "backtesting.log"
    error_log_file: str = "errors.log"
    
    # Log rotation
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Performance logging
    enable_performance_logging: bool = True
    performance_log_file: str = "performance.log"


@dataclass
class PerformanceSettings:
    """Performance and optimization settings"""
    
    # Parallel processing
    max_workers: int = 4
    use_multiprocessing: bool = True
    chunk_size: int = 1000
    
    # Memory management
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    enable_memory_monitoring: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Optimization
    enable_optimization: bool = True
    optimization_method: str = "genetic"  # genetic, bayesian, grid_search


@dataclass
class SecuritySettings:
    """Security and authentication settings"""
    
    # API keys
    encrypt_api_keys: bool = True
    encryption_key: str = ""
    
    # Access control
    require_authentication: bool = False
    session_timeout: int = 3600  # 1 hour
    
    # Rate limiting
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    
    # Data protection
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 365


@dataclass
class Settings:
    """Main application settings"""
    
    # Application info
    app_name: str = "Master Backtesting System"
    app_version: str = "1.0.0"
    app_description: str = "Advanced backtesting platform for trading strategies"
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug_mode: bool = True
    
    # Paths
    base_dir: str = str(Path(__file__).parent.parent)
    data_dir: str = "./data"
    results_dir: str = "./results"
    strategies_dir: str = "./strategies"
    logs_dir: str = "./logs"
    temp_dir: str = "./temp"
    
    # Database
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    
    # APIs
    api: APISettings = field(default_factory=APISettings)
    
    # Logging
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    
    # Performance
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    
    # Security
    security: SecuritySettings = field(default_factory=SecuritySettings)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        'ai_strategy_generation': True,
        'portfolio_optimization': True,
        'risk_management': True,
        'real_time_monitoring': False,
        'live_trading': False,
        'backtesting': True,
        'optimization': True,
        'reporting': True,
        'visualization': True
    })
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure all directories exist
        self._ensure_directories()
        
        # Set environment-specific defaults
        self._set_environment_defaults()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            self.results_dir,
            self.strategies_dir,
            self.logs_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _set_environment_defaults(self):
        """Set environment-specific default values"""
        if self.environment == "production":
            self.debug_mode = False
            self.logging.console_level = "WARNING"
            self.logging.file_level = "INFO"
            self.performance.max_workers = min(self.performance.max_workers, 8)
        
        elif self.environment == "staging":
            self.debug_mode = False
            self.logging.console_level = "INFO"
            self.logging.file_level = "DEBUG"
    
    def update(self, **kwargs):
        """Update settings with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
        
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert settings to JSON string"""
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(self.to_dict(), indent=indent, default=datetime_converter)
    
    def save_to_file(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """Save settings to file"""
        if file_path is None:
            file_path = DEFAULT_CONFIG_FILE
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            logging.error(f"Failed to save settings to {file_path}: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'Settings':
        """Load settings from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert string dates back to datetime objects
            for date_field in ['created_at', 'updated_at']:
                if date_field in data and isinstance(data[date_field], str):
                    data[date_field] = datetime.fromisoformat(data[date_field])
            
            return cls(**data)
        except Exception as e:
            logging.error(f"Failed to load settings from {file_path}: {e}")
            return cls()  # Return default settings
    
    @classmethod
    def load_from_env(cls) -> 'Settings':
        """Load settings from environment variables"""
        settings = cls()
        
        # Load from environment variables
        env_mappings = {
            'BACKTESTING_ENVIRONMENT': 'environment',
            'BACKTESTING_DEBUG': 'debug_mode',
            'BACKTESTING_DATA_DIR': 'data_dir',
            'BACKTESTING_RESULTS_DIR': 'results_dir',
            'BACKTESTING_STRATEGIES_DIR': 'strategies_dir',
            'BACKTESTING_LOGS_DIR': 'logs_dir',
            'BACKTESTING_TEMP_DIR': 'temp_dir',
            'BACKTESTING_MAX_WORKERS': 'performance.max_workers',
            'BACKTESTING_DATABASE_TYPE': 'database.database_type',
            'BACKTESTING_DATABASE_HOST': 'database.host',
            'BACKTESTING_DATABASE_PORT': 'database.port',
            'BACKTESTING_DATABASE_NAME': 'database.database_name',
            'BACKTESTING_ALPHA_VANTAGE_KEY': 'api.alpha_vantage_key',
            'BACKTESTING_POLYGON_KEY': 'api.polygon_key',
            'BACKTESTING_CONSOLE_LOG_LEVEL': 'logging.console_level',
            'BACKTESTING_FILE_LOG_LEVEL': 'logging.file_level'
        }
        
        for env_var, setting_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                cls._set_nested_value(settings, setting_path, value)
        
        return settings
    
    @staticmethod
    def _set_nested_value(obj, path: str, value: Any):
        """Set a nested value using dot notation"""
        parts = path.split('.')
        current = obj
        
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return
        
        # Convert value to appropriate type
        attr = getattr(current, parts[-1])
        if isinstance(attr, bool):
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(attr, int):
            value = int(value)
        elif isinstance(attr, float):
            value = float(value)
        
        setattr(current, parts[-1], value)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance"""
    global _settings
    
    if _settings is None:
        # Try to load from file first
        if DEFAULT_CONFIG_FILE.exists():
            _settings = Settings.load_from_file(DEFAULT_CONFIG_FILE)
        else:
            # Load from environment variables
            _settings = Settings.load_from_env()
            
            # Save default settings to file
            _settings.save_to_file()
    
    return _settings


def load_settings_from_file(file_path: Union[str, Path]) -> Settings:
    """Load settings from a specific file"""
    global _settings
    _settings = Settings.load_from_file(file_path)
    return _settings


def save_settings_to_file(settings: Settings, file_path: Optional[Union[str, Path]] = None) -> bool:
    """Save settings to a specific file"""
    return settings.save_to_file(file_path)


def reload_settings() -> Settings:
    """Reload settings from file"""
    global _settings
    _settings = None
    return get_settings()


# Initialize settings when module is imported
if __name__ == "__main__":
    # Create and save default settings
    settings = get_settings()
    print(f"Settings loaded: {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug_mode}")
    print(f"Data directory: {settings.data_dir}")
    print(f"Results directory: {settings.results_dir}")
