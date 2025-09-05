"""
Backtesting Engines Package

This package contains all the backtesting engine implementations, each incorporating
sophisticated testing and validation methods for identifying the best strategies.
"""

from .simple_engine import SimpleEngine
from .advanced_engine import AdvancedEngine, AdvancedEngineConfig, AdvancedBacktestResult
from .permutation_engine import PermutationEngine, PermutationEngineConfig, PermutationTestResult
from .risk_engine import RiskEngine, RiskEngineConfig, RiskBacktestResult
from .portfolio_engine import PortfolioEngine
from .ai_engine import AIEngine

__all__ = [
    'SimpleEngine',
    'AdvancedEngine',
    'AdvancedEngineConfig', 
    'AdvancedBacktestResult',
    'PermutationEngine',
    'PermutationEngineConfig',
    'PermutationTestResult',
    'RiskEngine',
    'RiskEngineConfig',
    'RiskBacktestResult',
    'PortfolioEngine',
    'AIEngine'
]
