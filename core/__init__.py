"""
Core Backtesting Framework

This package contains the core components of the backtesting system including
base classes, engines, strategies, data handlers, risk managers, and utilities.
"""

from .base import (
    BaseEngine,
    BaseStrategy,
    BaseDataHandler,
    BaseRiskManager,
    EngineConfig,
    StrategyConfig,
    DataConfig,
    RiskConfig,
    BacktestResult,
    Signal,
    Trade,
    PositionRisk,
    PortfolioRisk
)

from .engines import (
    SimpleEngine,
    AdvancedEngine,
    PortfolioEngine,
    PermutationEngine,
    RiskEngine,
    AIEngine
)

from .strategies import (
    BaseStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    ScalpingStrategy
)

from .data import (
    LocalDataHandler,
    APIDataHandler,
    DatabaseDataHandler
)

from .risk_management import (
    BasicRiskManager,
    AdvancedRiskManager,
    PortfolioRiskManager
)

from .performance import (
    PerformanceMetrics,
    PerformanceAnalyzer,
    ReportGenerator,
    ChartGenerator
)

from .utils import (
    ConfigManager,
    Logger,
    FileUtils,
    MathUtils
)

__all__ = [
    # Base classes
    'BaseEngine',
    'BaseStrategy', 
    'BaseDataHandler',
    'BaseRiskManager',
    
    # Configurations
    'EngineConfig',
    'StrategyConfig',
    'DataConfig',
    'RiskConfig',
    
    # Data structures
    'BacktestResult',
    'Signal',
    'Trade',
    'PositionRisk',
    'PortfolioRisk',
    
    # Engines
    'SimpleEngine',
    'AdvancedEngine',
    'PortfolioEngine',
    'PermutationEngine',
    'RiskEngine',
    'AIEngine',
    
    # Strategies
    'MomentumStrategy',
    'MeanReversionStrategy',
    'ScalpingStrategy',
    
    # Data handlers
    'LocalDataHandler',
    'APIDataHandler',
    'DatabaseDataHandler',
    
    # Risk managers
    'BasicRiskManager',
    'AdvancedRiskManager',
    'PortfolioRiskManager',
    
    # Performance tools
    'PerformanceMetrics',
    'PerformanceAnalyzer',
    'ReportGenerator',
    'ChartGenerator',
    
    # Utilities
    'ConfigManager',
    'Logger',
    'FileUtils',
    'MathUtils'
]

# Version information
__version__ = "1.0.0"
__author__ = "Andre"
__description__ = "Master Backtesting Architecture - Core Framework"
