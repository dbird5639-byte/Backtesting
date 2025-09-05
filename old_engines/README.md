# Backtesting Engines

A comprehensive suite of backtesting engines for trading strategy analysis and validation.

## Overview

This package provides multiple specialized backtesting engines that can be used individually or in a pipeline for comprehensive strategy testing. Each engine is designed to provide different types of analysis and validation.

Key improvements (what you can expect):
- Automatic truncation of oversized datasets so long CSVs no longer fail
- Parallel per-file backtesting using multiple CPU cores
- Automatic resume from the latest results directory per engine
- Cleaner alpha/statistics summaries with fewer external dependencies

## Engine Types

### 1. Basic Engine (`BasicEngine`)
**Purpose**: Fundamental backtesting with comprehensive statistics and quality assessment.

**Features**:
- Standard backtesting with full statistics
- Significance testing against random data
- Quality assessment scoring
- Configurable quality gate via `min_quality_score` to filter poor runs in summaries
- Comprehensive result reporting
- Parallel execution and auto-resume

**Configuration**:
```python
from engines import BasicEngine, BasicEngineConfig

config = BasicEngineConfig(
    run_significance_tests=True,
    n_permutations=10,
    min_quality_score=2.0
)
engine = BasicEngine(config)
results = engine.run()
```

### 2. Risk-Managed Engine (`RiskManagedEngine`)
**Purpose**: Advanced backtesting with comprehensive risk management features.

**Features**:
- Position sizing controls
- Stop-loss and take-profit management
- Trailing stops
- Drawdown protection
- Consecutive loss limits
- Risk-adjusted performance metrics
- Walkforward optimization for risk parameters
- Parallel execution and auto-resume

**Configuration**:
```python
from engines import RiskManagedEngine, RiskManagedEngineConfig

config = RiskManagedEngineConfig(
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    trailing_stop_pct=0.015,
    position_size_pct=0.10,
    max_consecutive_losses=5,
    max_drawdown_pct=0.20
)
engine = RiskManagedEngine(config)
results = engine.run()
```

### 3. Statistical Engine (`StatisticalEngine`)
**Purpose**: Advanced statistical analysis and validation of trading strategies.

**Features**:
- Permutation tests for significance
- Bootstrap confidence intervals
- Monte Carlo simulations
- Market regime analysis
- Statistical validation metrics
- Parallel execution and auto-resume

**Configuration**:
```python
from engines import StatisticalEngine, StatisticalEngineConfig

config = StatisticalEngineConfig(
    n_permutations=50,
    confidence_level=0.95,
    significance_level=0.05,
    enable_regime_analysis=True
)
engine = StatisticalEngine(config)
results = engine.run()
```

### 4. Walkforward Engine (`WalkforwardEngine`)
**Purpose**: Time-series cross-validation with market regime detection.

**Features**:
- Walkforward analysis (train/test splits)
- Market regime detection
- Regime-specific performance analysis
- Time-series validation
- Out-of-sample testing
- Parallel execution and auto-resume

**Configuration**:
```python
from engines import WalkforwardEngine, WalkforwardConfig

config = WalkforwardConfig(
    train_size=1000,
    test_size=200,
    step_size=200,
    enable_regime_analysis=True
)
engine = WalkforwardEngine(config)
results = engine.run()
```

### 5. Alpha Engine (`AlphaEngine`)
**Purpose**: Alpha detection, decay analysis, and signal strength assessment.

**Features**:
- Alpha period detection
- Alpha decay analysis
- Signal strength metrics
- Signal effectiveness analysis
- Decay cycle detection
- Parallel execution and auto-resume

**Configuration**:
```python
from engines import AlphaEngine, AlphaEngineConfig

config = AlphaEngineConfig(
    min_alpha_threshold=0.01,
    window_size=50,
    enable_alpha_detection=True,
    enable_decay_analysis=True,
    enable_signal_analysis=True
)
engine = AlphaEngine(config)
results = engine.run()
```

### 6. Pipeline Engine (`PipelineEngine`)

### 7. Epoch Engine (`EpochEngine`)
Purpose: Repeated re-sampling evaluations across multiple epochs to assess stability/robustness.

Features:
- Epoch-wise repeated backtests per dataset
- Aggregates mean/std/per-epoch metrics
- Skips existing results like other engines

### 8. ML Engine (`MLEngine`)
Purpose: Apply simple ML baselines (e.g., logistic regression, random forest) to predict next-bar direction from engineered features and assess learnability/edge.

Features:
- Lightweight feature generation (returns, momentum, volatility, rolling stats)
- Train/test split, k-fold CV option
- Metrics: AUC, accuracy, precision/recall, information coefficient
- Per-strategy evaluation using winners folder routing
**Purpose**: Run multiple engines in sequence and combine their results.

**Features**:
- Sequential engine execution
- Cross-engine analysis
- Combined result reporting
- Strategy consistency analysis
- Best performing engine identification

**Configuration**:
```python
from engines import PipelineEngine, PipelineEngineConfig

config = PipelineEngineConfig(
    engines_to_run=['basic', 'risk_managed', 'statistical'],
    combine_results=True,
    save_combined_results=True
)
engine = PipelineEngine(config)
results = engine.run()
```

## Directory Structure

```
Backtesting/
├── engines/
│   ├── __init__.py
│   ├── base_engine.py
│   ├── basic_engine.py
│   ├── risk_managed_engine.py
│   ├── statistical_engine.py
│   ├── walkforward_engine.py
│   ├── alpha_engine.py
│   ├── pipeline_engine.py
│   ├── example_usage.py
│   └── README.md
├── strategies/
│   └── (your strategy files)
├── Data/
│   └── Hyperliquid/
│       └── (your data files)
└── results/
    └── (output files)
```

## Usage Examples

### Basic Usage

```python
from engines import BasicEngine

# Use default configuration
engine = BasicEngine()
results = engine.run()
```

### Custom Configuration

```python
from engines import BasicEngine, BasicEngineConfig

config = BasicEngineConfig(
    data_path=r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\Data\Hyperliquid",
    strategies_path=r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\strategies",
    results_path=r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\results",
    initial_cash=10000,
    commission=0.002,
    run_significance_tests=True
)

engine = BasicEngine(config)
results = engine.run()
```

### Pipeline Usage

```python
from engines import PipelineEngine, PipelineEngineConfig

config = PipelineEngineConfig(
    engines_to_run=['basic', 'risk_managed', 'statistical'],
    combine_results=True
)

engine = PipelineEngine(config)
results = engine.run()
```

### Targeted Analysis

```python
from engines import PipelineEngine

engine = PipelineEngine()

# Run pipeline on specific strategies and data
results = engine.run_targeted_pipeline(
    target_strategies=['my_strategy'],
    target_data_files=['BTCUSDT_1m']
)
```

## Configuration Options

### Base Configuration (EngineConfig)
- `data_path`: Path to data files
- `strategies_path`: Path to strategy files
- `results_path`: Path to save results
- `initial_cash`: Starting capital
- `commission`: Trading commission
- `backtest_timeout`: Timeout for backtests
- `save_json`: Save results as JSON
- `save_csv`: Save results as CSV
- `log_level`: Logging level
- `min_data_points`: Minimum rows required
- `max_data_points`: Maximum rows allowed
- `truncate_excess_data`: If True, truncate data instead of error when above max
- `truncate_side`: "tail" (keep most recent rows, default) or "head"

### Engine-Specific Configurations

Each engine has its own configuration class with specialized parameters:

- **BasicEngineConfig**: Significance testing parameters
  - `parallel_workers`: Number of threads for per-file backtests (default 1)
  - `prefer_existing_results_dir`: Resume from latest `basic_backtest_*` folder (default True)
  - `results_subdir_prefix`: Results folder prefix (default "basic_backtest")

- **RiskManagedEngineConfig**: Risk management parameters
  - `parallel_workers`, `prefer_existing_results_dir`, `results_subdir_prefix` (default "risk_managed_backtest")

- **StatisticalEngineConfig**: Statistical analysis parameters
  - `n_bootstrap_samples`, `n_monte_carlo_simulations`
  - `run_permutation_tests`, `run_bootstrap_tests`, `run_monte_carlo_tests`, `enable_regime_analysis`
  - `parallel_workers`, `prefer_existing_results_dir`, `results_subdir_prefix` (default "statistical_backtest")

- **WalkforwardConfig**: Walkforward analysis parameters
  - `parallel_workers`, `prefer_existing_results_dir`, `results_subdir_prefix` (default "walkforward_performance_analysis")

- **AlphaEngineConfig**: Alpha analysis parameters
  - `parallel_workers`, `prefer_existing_results_dir`, `results_subdir_prefix` (default "alpha_analysis")

- **PipelineEngineConfig**: Pipeline execution parameters

## Output Structure

Each engine generates results in the following structure:

```
results/
├── engine_name_timestamp/
│   ├── strategy_name/
│   │   ├── data_file.json
│   │   └── ...
│   ├── all_results.json
│   ├── summary_stats.json
│   └── engine_specific_summary.json
```

## Key Features

### 1. Robust Error Handling
- Graceful shutdown on interruption
- Comprehensive logging
- Error recovery and reporting

### 2. Performance Optimization
- Skip existing results option
- Configurable timeouts
- Memory-efficient processing
- Parallel per-file execution with configurable workers
- Automatic resume using latest engine-specific results directory

### 3. Comprehensive Analysis
- Multiple statistical metrics
- Quality assessment scoring
- Cross-engine validation

### 4. Flexible Configuration
- Configurable paths
- Adjustable parameters
- Engine-specific settings

### 5. Pipeline Capabilities
- Sequential engine execution
- Combined result analysis
- Cross-engine consistency checking

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- backtesting (for strategy backtesting)
- SciPy is optional (no longer required by alpha engine)

## Installation

1. Ensure all required packages are installed:
```bash
pip install pandas numpy scikit-learn backtesting
# optional
pip install scipy
```

2. Place your strategy files in the `strategies/` directory
3. Place your data files in the `Data/Hyperliquid/` directory
4. Run the engines as needed

## Example Script

Run the example script to test all engines:

```bash
python engines/example_usage.py
```

This will demonstrate the usage of all engines and check your setup.

## Notes

- All engines use the same base configuration for paths and basic settings
- Results are saved in JSON format by default
- Each engine can be run independently or as part of a pipeline
- The pipeline engine provides cross-engine analysis capabilities
- All engines support graceful shutdown on interruption (Ctrl+C)
- The `run_engines.py` entrypoint sets `parallel_workers` to your CPU core count by default

## Recommended order to run engines

1) Basic Engine — establish a baseline and sanity-check stats/significance.
2) Risk-Managed Engine — apply realistic risk controls and (optionally) walkforward parameter checks.
3) Walkforward Engine — validate out-of-sample performance and regime behavior (no parameter tuning).
4) Statistical Engine — deeper significance/robustness via permutation, bootstrap, Monte Carlo, regimes.
5) Alpha Engine — analyze signal strength, alpha periods, and decay characteristics.

Tip: You can run them all via the Pipeline Engine, or selectively via `run_engines.py` prompts.

## Support

For issues or questions, check the example usage script and ensure all paths are correctly configured for your system. 