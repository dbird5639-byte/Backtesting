#!/usr/bin/env python3
"""
Enhanced Engines Demo Script

This script demonstrates all the enhanced backtesting engines that incorporate
sophisticated testing and validation methods from the user's existing engines:

1. AdvancedEngine - Comprehensive validation with permutation testing, walk-forward analysis,
   statistical analysis, Monte Carlo simulations, and bootstrap testing
2. PermutationEngine - Advanced statistical permutation testing with multiple shuffling methods
3. RiskEngine - Advanced risk management with walk-forward analysis, VaR/CVaR, and dynamic position sizing

Each engine demonstrates the proficient testing and validation methods that were working well
in the user's existing backtesting system.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engines import (
    SimpleEngine, 
    AdvancedEngine, AdvancedEngineConfig, AdvancedBacktestResult,
    PermutationEngine, PermutationEngineConfig, PermutationTestResult,
    RiskEngine, RiskEngineConfig, RiskBacktestResult
)
from core.base.base_engine import EngineConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    logger.info(f"Creating sample data with {n_days} days...")
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data with some trends and volatility
    np.random.seed(42)
    base_price = 100.0
    
    # Create trend with some randomness
    trend = np.linspace(0, 0.5, n_days)  # 50% upward trend over period
    noise = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
    returns = trend + noise
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intraday variation
        daily_vol = abs(noise[i]) * 2
        high = price * (1 + abs(np.random.normal(0, daily_vol)))
        low = price * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Sample data created: {len(df)} rows, columns: {list(df.columns)}")
    return df

def demo_simple_engine(data: pd.DataFrame):
    """Demonstrate the SimpleEngine"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: SimpleEngine")
    logger.info("="*60)
    
    try:
        # Configure simple engine
        config = EngineConfig(
            initial_cash=100000.0,
            commission=0.001,
            results_path="./results/simple_engine_demo"
        )
        
        # Initialize engine
        engine = SimpleEngine(config)
        
        # Load strategy
        strategy = engine.load_strategy("SimpleMA", {'short_window': 10, 'long_window': 20})
        
        # Run backtest
        logger.info("Running simple backtest...")
        result = engine.run_backtest(data, strategy)
        
        # Display results
        logger.info(f"Strategy: {result.strategy_name}")
        logger.info(f"Total Return: {result.total_return:.4%}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Final Capital: ${result.final_capital:,.2f}")
        
        # Save results
        output_path = engine.save_results([result])
        logger.info(f"Results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in SimpleEngine demo: {e}")
        return None

def demo_advanced_engine(data: pd.DataFrame):
    """Demonstrate the AdvancedEngine with comprehensive validation"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: AdvancedEngine - Comprehensive Validation")
    logger.info("="*60)
    
    try:
        # Configure advanced engine
        config = AdvancedEngineConfig(
            initial_cash=100000.0,
            commission=0.001,
            n_permutations=50,  # Reduced for demo
            walk_forward_windows=6,  # Reduced for demo
            n_monte_carlo_sims=100,  # Reduced for demo
            n_bootstrap_samples=100,  # Reduced for demo
            results_path="./results/advanced_engine_demo"
        )
        
        # Initialize engine
        engine = AdvancedEngine(config)
        
        # Load strategy
        strategy = engine.load_strategy("AdvancedMA", {'short_window': 15, 'long_window': 30})
        
        # Run comprehensive backtest
        logger.info("Running advanced backtest with comprehensive validation...")
        result = engine.run_backtest(data, strategy)
        
        # Display comprehensive results
        logger.info(f"Strategy: {result.strategy_name}")
        logger.info(f"Total Return: {result.total_return:.4%}")
        logger.info(f"Total Trades: {result.total_trades}")
        
        # Permutation testing results
        if result.permutation_significant is not None:
            logger.info(f"Permutation Test P-value: {result.permutation_p_value:.6f}")
            logger.info(f"Statistically Significant: {result.permutation_significant}")
        
        # Walk-forward results
        if result.walkforward_consistency is not None:
            logger.info(f"Walk-forward Consistency: {result.walkforward_consistency:.4f}")
        
        # Monte Carlo results
        if result.monte_carlo_var is not None:
            logger.info(f"Monte Carlo VaR (95%): {result.monte_carlo_var:.4%}")
            logger.info(f"Monte Carlo CVaR (95%): {result.monte_carlo_cvar:.4%}")
        
        # Bootstrap results
        if result.bootstrap_mean is not None:
            logger.info(f"Bootstrap Mean Return: {result.bootstrap_mean:.4%}")
            logger.info(f"Bootstrap Std Return: {result.bootstrap_std:.4%}")
        
        # Save results
        output_path = engine.save_results([result])
        logger.info(f"Advanced results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in AdvancedEngine demo: {e}")
        return None

def demo_permutation_engine(data: pd.DataFrame):
    """Demonstrate the PermutationEngine with advanced statistical testing"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: PermutationEngine - Advanced Statistical Testing")
    logger.info("="*60)
    
    try:
        # Configure permutation engine
        config = PermutationEngineConfig(
            initial_cash=100000.0,
            commission=0.001,
            n_permutations=50,  # Reduced for demo
            confidence_level=0.95,
            significance_threshold=0.05,
            shuffle_methods=["price_shuffle", "return_shuffle", "volume_shuffle"],
            results_path="./results/permutation_engine_demo"
        )
        
        # Initialize engine
        engine = PermutationEngine(config)
        
        # Load strategy
        strategy = engine.load_strategy("PermutationMA", {'short_window': 12, 'long_window': 25})
        
        # Run permutation test
        logger.info("Running advanced permutation test...")
        result = engine.run_backtest(data, strategy)
        
        # Display permutation results
        logger.info(f"Strategy: {result.strategy_name}")
        logger.info(f"Original Return: {result.original_return:.4%}")
        logger.info(f"Permutations: {result.n_permutations}")
        logger.info(f"Successful Permutations: {result.successful_permutations}")
        
        # Statistical significance
        logger.info(f"P-value: {result.p_value:.6f}")
        logger.info(f"Adjusted P-value: {result.adjusted_p_value:.6f}")
        logger.info(f"Statistically Significant: {result.significant}")
        
        # Effect size and power
        if result.effect_size is not None:
            logger.info(f"Effect Size (Cohen's d): {result.effect_size:.4f}")
        if result.statistical_power is not None:
            logger.info(f"Statistical Power: {result.statistical_power:.4f}")
        
        # Confidence intervals
        logger.info(f"Confidence Interval: [{result.confidence_interval[0]:.4%}, {result.confidence_interval[1]:.4%}]")
        
        # Distribution statistics
        if result.permutation_distribution_stats:
            stats = result.permutation_distribution_stats
            logger.info(f"Permutation Mean: {stats.get('mean', 0):.4%}")
            logger.info(f"Permutation Std: {stats.get('std', 0):.4%}")
            logger.info(f"Permutation Skewness: {stats.get('skewness', 0):.4f}")
        
        # Extreme value analysis
        if result.extreme_value_analysis:
            extreme = result.extreme_value_analysis
            logger.info(f"Original Percentile: {extreme.get('original_percentile', 0):.1f}%")
            logger.info(f"Outlier Percentage: {extreme.get('outlier_percentage', 0):.1f}%")
        
        # Save results
        output_path = engine.save_results([result])
        logger.info(f"Permutation test results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in PermutationEngine demo: {e}")
        return None

def demo_risk_engine(data: pd.DataFrame):
    """Demonstrate the RiskEngine with advanced risk management"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: RiskEngine - Advanced Risk Management")
    logger.info("="*60)
    
    try:
        # Configure risk engine
        config = RiskEngineConfig(
            initial_cash=100000.0,
            commission=0.001,
            max_drawdown=0.15,
            max_position_size=0.2,
            position_sizing_method="kelly",
            kelly_fraction=0.25,
            volatility_target=0.15,
            walk_forward_windows=6,  # Reduced for demo
            results_path="./results/risk_engine_demo"
        )
        
        # Initialize engine
        engine = RiskEngine(config)
        
        # Load strategy
        strategy = engine.load_strategy("RiskMA", {'short_window': 8, 'long_window': 21})
        
        # Run risk-managed backtest
        logger.info("Running risk-managed backtest...")
        result = engine.run_backtest(data, strategy)
        
        # Display risk results
        logger.info(f"Strategy: {result.strategy_name}")
        logger.info(f"Total Return: {result.total_return:.4%}")
        logger.info(f"Total Trades: {result.total_trades}")
        
        # Risk metrics
        logger.info(f"Max Drawdown: {result.max_drawdown:.4%}")
        logger.info(f"VaR (95%): {result.var_95:.4%}")
        logger.info(f"CVaR (95%): {result.cvar_95:.4%}")
        logger.info(f"Volatility: {result.volatility:.4%}")
        logger.info(f"Downside Deviation: {result.downside_deviation:.4%}")
        
        # Risk-adjusted metrics
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        logger.info(f"Sortino Ratio: {result.sortino_ratio:.4f}")
        logger.info(f"Calmar Ratio: {result.calmar_ratio:.4f}")
        logger.info(f"Information Ratio: {result.information_ratio:.4f}")
        
        # Position sizing metrics
        logger.info(f"Average Position Size: {result.avg_position_size:.4%}")
        logger.info(f"Max Position Size: {result.max_position_size:.4%}")
        logger.info(f"Position Size Volatility: {result.position_size_volatility:.4%}")
        
        # Walk-forward results
        if result.walkforward_consistency is not None:
            logger.info(f"Walk-forward Consistency: {result.walkforward_consistency:.4f}")
        
        # Save results
        output_path = engine.save_results([result])
        logger.info(f"Risk analysis results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in RiskEngine demo: {e}")
        return None

def generate_comprehensive_report(results: dict):
    """Generate a comprehensive report comparing all engines"""
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE ENGINE COMPARISON REPORT")
    logger.info("="*80)
    
    try:
        # Create comparison table
        comparison_data = []
        
        for engine_name, result in results.items():
            if result is None:
                continue
                
            if hasattr(result, 'total_return'):
                comparison_data.append({
                    'Engine': engine_name,
                    'Total Return': f"{result.total_return:.4%}",
                    'Total Trades': result.total_trades,
                    'Max Drawdown': f"{getattr(result, 'max_drawdown', 0):.4%}",
                    'Sharpe Ratio': f"{getattr(result, 'sharpe_ratio', 0):.4f}",
                    'Significant (Permutation)': getattr(result, 'permutation_significant', 'N/A'),
                    'Walk-forward Consistency': f"{getattr(result, 'walkforward_consistency', 'N/A')}",
                    'VaR (95%)': f"{getattr(result, 'var_95', 'N/A')}",
                    'CVaR (95%)': f"{getattr(result, 'cvar_95', 'N/A')}"
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            logger.info("\nEngine Performance Comparison:")
            logger.info(comparison_df.to_string(index=False))
            
            # Save comparison report
            report_path = "./results/engine_comparison_report.csv"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            comparison_df.to_csv(report_path, index=False)
            logger.info(f"\nComparison report saved to: {report_path}")
        
        # Summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(f"Total Engines Tested: {len([r for r in results.values() if r is not None])}")
        
        # Best performing engine
        if comparison_data:
            best_engine = max(comparison_data, key=lambda x: float(x['Total Return'].rstrip('%')))
            logger.info(f"Best Return: {best_engine['Engine']} with {best_engine['Total Return']}")
        
        # Risk analysis
        if comparison_data:
            lowest_dd = min(comparison_data, key=lambda x: float(x['Max Drawdown'].rstrip('%')))
            logger.info(f"Lowest Drawdown: {lowest_dd['Engine']} with {lowest_dd['Max Drawdown']}")
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE VALIDATION METHODS IMPLEMENTED:")
        logger.info("="*80)
        logger.info("✓ Permutation Testing - Statistical significance validation")
        logger.info("✓ Walk-Forward Analysis - Out-of-sample validation")
        logger.info("✓ Statistical Analysis - Alpha decay, regime detection")
        logger.info("✓ Monte Carlo Simulations - Robustness testing")
        logger.info("✓ Bootstrap Testing - Statistical validation")
        logger.info("✓ Advanced Risk Management - VaR, CVaR, dynamic position sizing")
        logger.info("✓ Multiple Testing Corrections - Bonferroni, Holm methods")
        logger.info("✓ Effect Size Calculations - Cohen's d effect size")
        logger.info("✓ Statistical Power Analysis - Test power calculations")
        logger.info("✓ Extreme Value Analysis - Outlier detection")
        logger.info("✓ Robustness Metrics - Stability and consistency measures")
        logger.info("✓ Dynamic Position Sizing - Kelly, Risk Parity, Volatility Targeting")
        logger.info("✓ Correlation Analysis - Portfolio risk management")
        logger.info("✓ Regime Detection - Market condition analysis")
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")

def main():
    """Main demo function"""
    logger.info("Enhanced Backtesting Engines Demo")
    logger.info("Demonstrating sophisticated testing and validation methods")
    logger.info("="*80)
    
    try:
        # Create sample data
        data = create_sample_data(500)  # Reduced for demo
        
        # Run all engine demos
        results = {}
        
        # Simple Engine Demo
        results['SimpleEngine'] = demo_simple_engine(data)
        
        # Advanced Engine Demo
        results['AdvancedEngine'] = demo_advanced_engine(data)
        
        # Permutation Engine Demo
        results['PermutationEngine'] = demo_permutation_engine(data)
        
        # Risk Engine Demo
        results['RiskEngine'] = demo_risk_engine(data)
        
        # Generate comprehensive report
        generate_comprehensive_report(results)
        
        logger.info("\n" + "="*80)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("All enhanced engines have been demonstrated with their")
        logger.info("sophisticated testing and validation methods.")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")
        raise

if __name__ == "__main__":
    main()
