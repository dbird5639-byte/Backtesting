# Backtesting Engines

This directory contains the latest, improved backtesting engines based on successful patterns from the old_engines directory.

## Available Engines

### 1. Core Engine (`core_engine.py`) ⭐⭐⭐⭐⭐
- **Purpose**: Fundamental backtesting with comprehensive statistics and quality assessment
- **Features**: 
  - Parallel processing with ThreadPoolExecutor
  - Quality assessment scoring system
  - Significance testing with permutation tests
  - Robust error handling and graceful shutdown
  - Auto-resume functionality
  - Data truncation for oversized datasets
  - **Status**: ✅ Optimized with old_engines patterns

### 2. Risk Engine (`risk_engine.py`) ⭐⭐⭐⭐⭐
- **Purpose**: Advanced backtesting with comprehensive risk management features
- **Features**:
  - Advanced risk management with position sizing
  - Walkforward optimization for parameter tuning
  - Kelly Criterion position sizing
  - Multiple risk metrics (drawdown, consecutive losses)
  - Risk-adjusted performance analysis
  - Parameter optimization with grid search
  - **Status**: ✅ Optimized with old_engines patterns

### 3. Statistical Engine (`statistical_engine.py`) ⭐⭐⭐⭐⭐
- **Purpose**: Advanced statistical analysis and validation of trading strategies
- **Features**:
  - Permutation tests for significance validation
  - Bootstrap analysis for confidence intervals
  - Monte Carlo simulations including GBM
  - Market regime detection with K-means clustering
  - Statistical validation metrics
  - Comprehensive analysis reporting
  - **Status**: ✅ Optimized with old_engines patterns

### 4. Pipeline Engine (`pipeline_engine.py`) ⭐⭐⭐⭐⭐
- **Purpose**: Orchestrates multiple engines in sequence with comprehensive result analysis
- **Features**:
  - Sequential engine execution with proper ordering
  - Cross-engine result combination
  - Targeted analysis capabilities
  - Comprehensive reporting with summaries
  - Result aggregation across engines
  - Performance monitoring
  - **Status**: ✅ Optimized with old_engines patterns

### 5. Validation Engine (`validation_engine.py`) ⭐⭐⭐⭐⭐
- **Purpose**: Comprehensive validation with multiple testing methodologies
- **Features**:
  - Cross-validation with time series splits
  - Walk-forward analysis for out-of-sample testing
  - Bootstrap validation for confidence intervals
  - Permutation tests for significance
  - Overfitting detection and prevention
  - Performance degradation analysis
  - **Status**: ✅ Completely rewritten with old_engines patterns

### 6. Portfolio Engine (`portfolio_engine.py`) ⭐⭐⭐⭐
- **Purpose**: Multi-objective portfolio optimization and management
- **Features**:
  - Advanced optimization algorithms
  - Dynamic rebalancing strategies
  - Transaction cost modeling
  - Factor-based portfolio construction
  - Real-time portfolio monitoring
  - **Status**: 🔄 Ready for optimization

### 7. ML Engine (`ml_engine.py`) ⭐⭐⭐⭐
- **Purpose**: Machine learning-powered backtesting and strategy development
- **Features**:
  - Automated feature engineering
  - Multiple ML model training and selection
  - Cross-validation and hyperparameter optimization
  - Model performance tracking
  - Ensemble methods and stacking
  - **Status**: 🔄 Ready for optimization

### 8. Performance Engine (`performance_engine.py`) ⭐⭐⭐⭐
- **Purpose**: Advanced performance analytics and reporting
- **Features**:
  - Advanced performance metrics calculation
  - Performance attribution and decomposition
  - Benchmark comparison and analysis
  - Performance visualization and reporting
  - Performance forecasting and prediction
  - **Status**: 🔄 Ready for optimization

### 9. Regime Detection Engine (`regime_detection_engine.py`) ⭐⭐⭐⭐
- **Purpose**: Market regime identification and analysis
- **Features**:
  - Multiple regime detection algorithms
  - Advanced feature engineering
  - Regime transition analysis
  - Interactive visualizations
  - **Status**: 🔄 Ready for optimization

### 10. Enhanced Visualization Engine (`enhanced_visualization_engine.py`) ⭐⭐⭐⭐
- **Purpose**: Comprehensive visualization capabilities for backtesting results
- **Features**:
  - Interactive plotly charts
  - Heatmap generation
  - Regime overlay visualization
  - Strategy comparison charts
  - **Status**: 🔄 Ready for optimization

### 11. Fibonacci/Gann Engine (`fibonacci_gann_engine.py`) ⭐⭐⭐⭐
- **Purpose**: Advanced Fibonacci and Gann analysis
- **Features**:
  - Fibonacci retracements, extensions, and time zones
  - Gann squares, fans, and angle analysis
  - Advanced swing point detection
  - Interactive visualizations
  - **Status**: 🔄 Ready for optimization

## Usage

### Individual Engine Usage
```python
from Engines.core_engine import CoreEngine, EngineConfig
from Engines.risk_engine import RiskEngine, RiskEngineConfig
from Engines.statistical_engine import StatisticalEngine, StatisticalEngineConfig
from Engines.validation_engine import ValidationEngine, ValidationEngineConfig

# Run individual engine
config = EngineConfig()
engine = CoreEngine(config)
engine.run()
```

### Pipeline Usage
```python
from Engines.pipeline_engine import PipelineEngine, PipelineEngineConfig

config = PipelineEngineConfig()
engine = PipelineEngine(config)
engine.run()
```

### Validation Engine Usage
```python
from Engines.validation_engine import ValidationEngine, ValidationEngineConfig

config = ValidationEngineConfig()
engine = ValidationEngine(config)
engine.run()
```

## Configuration

Each engine has its own configuration class with specialized parameters. See individual engine files for detailed configuration options.

## Results

Results are saved to the `Results/` directory with timestamps and organized by engine type.

## Old Engines

Previous versions of engines have been moved to `storage/engines_storage/` for reference.
