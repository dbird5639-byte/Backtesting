"""
Configuration file for Advanced Test Engine

This file contains all the configuration parameters for:
- In-Sample Excellence Testing
- Permutation Testing
- Walk Forward Testing
- Strategy Parameters
"""

# Strategy Parameter Grids
DONCHIAN_PARAM_GRID = {
    'lookback_period': [10, 15, 20, 25, 30, 35, 40],
    'atr_period': [10, 14, 20, 25],
    'risk_per_trade': [0.01, 0.10, 0.20],
    'trailing_stop': [True, False],
    'use_atr_stops': [True, False]
}

# Enhanced Strategy Parameter Grid
ENHANCED_DONCHIAN_PARAM_GRID = {
    'lookback_period': [15, 20, 25, 30],
    'atr_period': [10, 14, 20],
    'risk_per_trade': [0.01, 0.10, 0.20],
    'volume_threshold': [1.1, 1.2, 1.3],
    'momentum_period': [5, 10, 15],
    'min_holding_period': [3, 5, 7]
}

# Permutation Test Configuration
PERMUTATION_CONFIG = {
    'n_permutations': 100,  # Number of permutations for in-sample test
    'wf_n_permutations': 10,  # Number of permutations for walk forward test
    'preserve_volatility_clustering': True,
    'preserve_serial_correlation': True,
    'volatility_regimes': 5,  # Number of volatility regimes for clustering
    'min_permutation_length': 100  # Minimum data points for permutation
}

# Walk Forward Configuration
WALK_FORWARD_CONFIG = {
    'train_size': 0.7,  # 70% for training
    'step_size': 30,  # Days to step forward
    'min_test_period': 10,  # Minimum test period length
    'max_test_period': 90,  # Maximum test period length
    'overlap_allowed': False,  # Allow overlapping test periods
    'min_trades_per_test': 5  # Minimum trades required per test period
}

# Performance Metrics Configuration
PERFORMANCE_CONFIG = {
    'risk_free_rate': 0.02,  # 2% risk-free rate for Sharpe calculation
    'benchmark_return': 0.08,  # 8% annual benchmark return
    'max_drawdown_threshold': 0.20,  # 20% maximum acceptable drawdown
    'sharpe_threshold': 0.5,  # Minimum acceptable Sharpe ratio
    'profit_factor_threshold': 1.2,  # Minimum acceptable profit factor
    'win_rate_threshold': 0.4  # Minimum acceptable win rate
}

# Statistical Significance Configuration
SIGNIFICANCE_CONFIG = {
    'p_value_threshold': 0.05,  # 5% significance level
    'bonferroni_correction': True,  # Apply Bonferroni correction for multiple tests
    'confidence_interval': 0.95,  # 95% confidence interval
    'bootstrap_samples': 1000,  # Number of bootstrap samples
    'permutation_test_method': 'exact'  # 'exact' or 'approximate'
}

# Data Processing Configuration
DATA_CONFIG = {
    'min_data_points': 500,  # Minimum data points required
    'max_missing_data': 0.05,  # Maximum 5% missing data allowed
    'outlier_threshold': 3.0,  # Standard deviations for outlier detection
    'resample_frequency': None,  # None for no resampling, or '1D', '1H', etc.
    'fill_method': 'forward'  # 'forward', 'backward', 'interpolate'
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_detailed_results': True,
    'save_plots': True,
    'save_permutation_distributions': True,
    'save_walk_forward_charts': True,
    'plot_format': 'png',  # 'png', 'pdf', 'svg'
    'dpi': 300,
    'figure_size': (12, 8)
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
    'log_file_prefix': 'advanced_test'
}

# Test Execution Configuration
EXECUTION_CONFIG = {
    'parallel_processing': True,  # Enable parallel processing
    'max_workers': 2,  # Maximum number of parallel workers
    'timeout_per_test': 43200,  # Timeout in seconds per test
    'memory_limit_gb': 8,  # Memory limit in GB
    'continue_on_error': True,  # Continue testing if one test fails
    'save_intermediate_results': True  # Save results after each test
}

# Strategy Quality Assessment Configuration
QUALITY_CONFIG = {
    'excellent_threshold': {
        'sharpe_ratio': 1.5,
        'calmar_ratio': 2.0,
        'profit_factor': 2.0,
        'win_rate': 0.6,
        'max_drawdown': 0.15
    },
    'good_threshold': {
        'sharpe_ratio': 1.0,
        'calmar_ratio': 1.5,
        'profit_factor': 1.5,
        'win_rate': 0.5,
        'max_drawdown': 0.20
    },
    'fair_threshold': {
        'sharpe_ratio': 0.5,
        'calmar_ratio': 1.0,
        'profit_factor': 1.2,
        'win_rate': 0.4,
        'max_drawdown': 0.25
    }
}

# Market Regime Detection Configuration
REGIME_CONFIG = {
    'volatility_regimes': 3,  # Number of volatility regimes
    'momentum_regimes': 2,  # Number of momentum regimes
    'regime_detection_method': 'kmeans',  # 'kmeans', 'gmm', 'hierarchical'
    'regime_min_duration': 20,  # Minimum duration for a regime
    'regime_transition_threshold': 0.1  # Threshold for regime transition
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_position_size': 0.1,  # Maximum 10% position size
    'max_portfolio_risk': 0.02,  # Maximum 2% portfolio risk
    'correlation_threshold': 0.7,  # Maximum correlation between positions
    'var_confidence': 0.95,  # Value at Risk confidence level
    'max_consecutive_losses': 5,  # Maximum consecutive losses
    'profit_taking_multiplier': 2.0  # Take profit at 2x risk
}

# Validation Configuration
VALIDATION_CONFIG = {
    'cross_validation_folds': 5,
    'time_series_cv': True,
    'blocking_factor': 10,  # For time series cross-validation
    'validation_metric': 'sharpe_ratio',  # Primary validation metric
    'secondary_metrics': ['calmar_ratio', 'profit_factor', 'max_drawdown']
} 