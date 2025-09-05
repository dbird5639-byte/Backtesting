"""
Base Classes Package

This package contains the abstract base classes that define the interfaces
for all backtesting components including engines, strategies, data handlers,
and risk managers.
"""

from .base_engine import BaseEngine
from .base_strategy import BaseStrategy
from .base_data_handler import BaseDataHandler
from .base_risk_manager import BaseRiskManager

__all__ = [
    'BaseEngine',
    'BaseStrategy', 
    'BaseDataHandler',
    'BaseRiskManager'
]
