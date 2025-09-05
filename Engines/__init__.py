# Backtesting Engines Package

"""
This package contains the latest, optimized backtesting engines based on successful patterns from old_engines.

Available Engines:
- Core Engine: Fundamental backtesting with quality assessment (Optimized)
- Risk Engine: Advanced risk management and walkforward optimization (Optimized)
- Statistical Engine: Statistical validation and regime analysis (Optimized)
- Pipeline Engine: Orchestrates multiple engines in sequence (Optimized)
- Validation Engine: Comprehensive validation with multiple testing methods (Optimized)
- Portfolio Engine: Multi-objective portfolio optimization (Ready for optimization)
- ML Engine: Machine learning-powered backtesting (Ready for optimization)
- Performance Engine: Advanced performance analytics (Ready for optimization)
- Regime Detection Engine: Market regime identification (Ready for optimization)
- Visualization Engine: Comprehensive visualization capabilities (Ready for optimization)
- Fibonacci/Gann Engine: Advanced technical analysis (Ready for optimization)
"""

# Import all engines
from .core_engine import CoreEngine, EngineConfig
from .risk_engine import RiskEngine, RiskEngineConfig
from .statistical_engine import StatisticalEngine, StatisticalEngineConfig
from .pipeline_engine import PipelineEngine, PipelineEngineConfig
from .validation_engine import ValidationEngine, ValidationEngineConfig
from .portfolio_engine import PortfolioEngine, PortfolioEngineConfig
from .ml_engine import MLEngine, MLEngineConfig
from .performance_engine import PerformanceEngine, PerformanceEngineConfig
from .regime_detection_engine import RegimeDetectionEngine, RegimeDetectionConfig
from .regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
from .regime_visualization_engine import RegimeVisualizationEngine, RegimeVisualizationConfig
from .enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
from .fibonacci_gann_engine import FibonacciGannEngine, FibonacciGannConfig

# Import optimized engines
from .optimized_portfolio_engine import OptimizedPortfolioEngine, PortfolioEngineConfig as OptimizedPortfolioConfig
from .optimized_ml_engine import OptimizedMLEngine, MLEngineConfig as OptimizedMLConfig
from .optimized_regime_engine import OptimizedRegimeEngine, RegimeEngineConfig

__all__ = [
    'CoreEngine', 'EngineConfig',
    'RiskEngine', 'RiskEngineConfig',
    'StatisticalEngine', 'StatisticalEngineConfig',
    'PipelineEngine', 'PipelineEngineConfig',
    'ValidationEngine', 'ValidationEngineConfig',
    'PortfolioEngine', 'PortfolioEngineConfig',
    'MLEngine', 'MLEngineConfig',
    'PerformanceEngine', 'PerformanceEngineConfig',
    'RegimeDetectionEngine', 'RegimeDetectionConfig',
    'RegimeOverlayEngine', 'RegimeOverlayConfig',
    'RegimeVisualizationEngine', 'RegimeVisualizationConfig',
    'EnhancedVisualizationEngine', 'VisualizationConfig',
    'FibonacciGannEngine', 'FibonacciGannConfig',
    # Optimized engines
    'OptimizedPortfolioEngine', 'OptimizedPortfolioConfig',
    'OptimizedMLEngine', 'OptimizedMLConfig',
    'OptimizedRegimeEngine', 'RegimeEngineConfig'
]