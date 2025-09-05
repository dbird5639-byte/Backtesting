"""
Research Methodology System
Based on AI project methodologies for systematic quantitative research
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from scipy import stats
from scipy.optimize import minimize
import json
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class ResearchPhase(Enum):
    """Research phases"""
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_DEVELOPMENT = "model_development"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"

class ResearchMethod(Enum):
    """Research methods"""
    STATISTICAL_ANALYSIS = "statistical_analysis"
    MACHINE_LEARNING = "machine_learning"
    TIME_SERIES_ANALYSIS = "time_series_analysis"
    REGIME_ANALYSIS = "regime_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    BACKTESTING = "backtesting"

@dataclass
class ResearchHypothesis:
    """Research hypothesis data structure"""
    id: str
    title: str
    description: str
    methodology: ResearchMethod
    expected_outcome: str
    confidence_level: float
    test_statistic: Optional[str] = None
    p_value_threshold: float = 0.05
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchResult:
    """Research result data structure"""
    hypothesis_id: str
    method: ResearchMethod
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    interpretation: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResearchMethodologySystem:
    """
    Comprehensive research methodology system based on AI project insights
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Research state
        self.hypotheses: Dict[str, ResearchHypothesis] = {}
        self.results: List[ResearchResult] = []
        self.research_log: List[Dict[str, Any]] = []
        
        # Research parameters
        self.significance_level = self.config.get('significance_level', 0.05)
        self.min_effect_size = self.config.get('min_effect_size', 0.1)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        self.logger.info("Research Methodology System initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'significance_level': 0.05,
            'min_effect_size': 0.1,
            'confidence_level': 0.95,
            'min_sample_size': 30,
            'max_p_value': 0.05,
            'correlation_threshold': 0.3,
            'regime_threshold': 0.05,
            'anomaly_threshold': 0.1
        }
    
    def create_hypothesis(self, title: str, description: str, methodology: ResearchMethod,
                         expected_outcome: str, confidence_level: float = 0.8) -> str:
        """Create a new research hypothesis"""
        hypothesis_id = f"hyp_{len(self.hypotheses) + 1:03d}"
        
        hypothesis = ResearchHypothesis(
            id=hypothesis_id,
            title=title,
            description=description,
            methodology=methodology,
            expected_outcome=expected_outcome,
            confidence_level=confidence_level
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        
        self.logger.info(f"Created hypothesis: {title}")
        return hypothesis_id
    
    def test_hypothesis(self, hypothesis_id: str, data: pd.DataFrame, 
                       **kwargs) -> ResearchResult:
        """Test a research hypothesis"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        self.logger.info(f"Testing hypothesis: {hypothesis.title}")
        
        # Route to appropriate test method
        if hypothesis.methodology == ResearchMethod.STATISTICAL_ANALYSIS:
            result = self._test_statistical_hypothesis(hypothesis, data, **kwargs)
        elif hypothesis.methodology == ResearchMethod.MACHINE_LEARNING:
            result = self._test_ml_hypothesis(hypothesis, data, **kwargs)
        elif hypothesis.methodology == ResearchMethod.TIME_SERIES_ANALYSIS:
            result = self._test_timeseries_hypothesis(hypothesis, data, **kwargs)
        elif hypothesis.methodology == ResearchMethod.REGIME_ANALYSIS:
            result = self._test_regime_hypothesis(hypothesis, data, **kwargs)
        elif hypothesis.methodology == ResearchMethod.CORRELATION_ANALYSIS:
            result = self._test_correlation_hypothesis(hypothesis, data, **kwargs)
        elif hypothesis.methodology == ResearchMethod.ANOMALY_DETECTION:
            result = self._test_anomaly_hypothesis(hypothesis, data, **kwargs)
        elif hypothesis.methodology == ResearchMethod.BACKTESTING:
            result = self._test_backtesting_hypothesis(hypothesis, data, **kwargs)
        else:
            raise ValueError(f"Unknown methodology: {hypothesis.methodology}")
        
        # Store result
        self.results.append(result)
        
        # Update hypothesis status
        hypothesis.status = "completed"
        hypothesis.results = result.metadata
        
        # Log research activity
        self._log_research_activity(hypothesis, result)
        
        return result
    
    def _test_statistical_hypothesis(self, hypothesis: ResearchHypothesis, 
                                   data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test statistical hypothesis"""
        # Extract variables from data
        x_var = kwargs.get('x_variable', 'close')
        y_var = kwargs.get('y_variable', 'returns')
        
        if x_var not in data.columns or y_var not in data.columns:
            raise ValueError(f"Required variables not found in data")
        
        x = data[x_var].dropna()
        y = data[y_var].dropna()
        
        # Align data
        common_index = x.index.intersection(y.index)
        x = x.loc[common_index]
        y = y.loc[common_index]
        
        # Perform statistical test
        if hypothesis.title.lower().find('correlation') != -1:
            # Correlation test
            correlation, p_value = stats.pearsonr(x, y)
            test_statistic = correlation
            effect_size = abs(correlation)
            
            interpretation = f"Correlation coefficient: {correlation:.3f}, p-value: {p_value:.3f}"
            
        elif hypothesis.title.lower().find('normality') != -1:
            # Normality test
            test_statistic, p_value = stats.shapiro(x)
            effect_size = 0.0  # Not applicable for normality test
            
            interpretation = f"Shapiro-Wilk test statistic: {test_statistic:.3f}, p-value: {p_value:.3f}"
            
        elif hypothesis.title.lower().find('stationarity') != -1:
            # Stationarity test (simplified)
            from statsmodels.tsa.stattools import adfuller
            test_statistic, p_value, _, _, _, _ = adfuller(x)
            effect_size = 0.0  # Not applicable for stationarity test
            
            interpretation = f"ADF test statistic: {test_statistic:.3f}, p-value: {p_value:.3f}"
            
        else:
            # Default: t-test
            test_statistic, p_value = stats.ttest_1samp(x, 0)
            effect_size = abs(x.mean() / x.std()) if x.std() > 0 else 0
            
            interpretation = f"One-sample t-test statistic: {test_statistic:.3f}, p-value: {p_value:.3f}"
        
        # Calculate confidence interval
        if len(x) > 1:
            ci = stats.t.interval(self.confidence_level, len(x)-1, 
                                loc=x.mean(), scale=stats.sem(x))
        else:
            ci = (0, 0)
        
        # Determine significance
        is_significant = p_value < self.significance_level
        
        # Generate recommendations
        recommendations = self._generate_statistical_recommendations(
            test_statistic, p_value, effect_size, is_significant
        )
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'sample_size': len(x),
                'x_variable': x_var,
                'y_variable': y_var,
                'x_mean': x.mean(),
                'x_std': x.std(),
                'y_mean': y.mean() if y_var in data.columns else None,
                'y_std': y.std() if y_var in data.columns else None
            }
        )
    
    def _test_ml_hypothesis(self, hypothesis: ResearchHypothesis, 
                          data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test machine learning hypothesis"""
        # This is a simplified ML test - in practice would use scikit-learn
        target_var = kwargs.get('target_variable', 'returns')
        feature_vars = kwargs.get('feature_variables', ['close'])
        
        if target_var not in data.columns:
            raise ValueError(f"Target variable {target_var} not found")
        
        # Create features
        features = data[feature_vars].dropna()
        target = data[target_var].dropna()
        
        # Align data
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        model = LinearRegression()
        model.fit(features, target)
        
        predictions = model.predict(features)
        r2 = r2_score(target, predictions)
        mse = mean_squared_error(target, predictions)
        
        # Calculate test statistic (F-statistic approximation)
        n = len(target)
        p = len(feature_vars)
        f_statistic = (r2 / p) / ((1 - r2) / (n - p - 1)) if n > p + 1 else 0
        
        # Calculate p-value (simplified)
        p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1) if n > p + 1 else 1.0
        
        effect_size = r2
        is_significant = p_value < self.significance_level
        
        interpretation = f"R²: {r2:.3f}, F-statistic: {f_statistic:.3f}, p-value: {p_value:.3f}"
        
        recommendations = self._generate_ml_recommendations(r2, f_statistic, p_value, is_significant)
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=f_statistic,
            p_value=p_value,
            confidence_interval=(0, 0),  # Not applicable for ML
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'r2_score': r2,
                'mse': mse,
                'n_features': p,
                'n_samples': n,
                'feature_importance': dict(zip(feature_vars, model.coef_))
            }
        )
    
    def _test_timeseries_hypothesis(self, hypothesis: ResearchHypothesis, 
                                  data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test time series hypothesis"""
        price_var = kwargs.get('price_variable', 'close')
        
        if price_var not in data.columns:
            raise ValueError(f"Price variable {price_var} not found")
        
        prices = data[price_var].dropna()
        returns = prices.pct_change().dropna()
        
        # Test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        ljung_box = acorr_ljungbox(returns, lags=10, return_df=True)
        test_statistic = ljung_box['lb_stat'].iloc[-1]
        p_value = ljung_box['lb_pvalue'].iloc[-1]
        
        # Calculate effect size (autocorrelation coefficient)
        effect_size = abs(returns.autocorr(lag=1)) if len(returns) > 1 else 0
        
        is_significant = p_value < self.significance_level
        
        interpretation = f"Ljung-Box test statistic: {test_statistic:.3f}, p-value: {p_value:.3f}"
        
        recommendations = self._generate_timeseries_recommendations(
            test_statistic, p_value, effect_size, is_significant
        )
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=(0, 0),  # Not applicable for Ljung-Box test
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'n_observations': len(returns),
                'autocorr_lag1': returns.autocorr(lag=1) if len(returns) > 1 else 0,
                'volatility': returns.std() * np.sqrt(252),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
        )
    
    def _test_regime_hypothesis(self, hypothesis: ResearchHypothesis, 
                              data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test regime hypothesis"""
        price_var = kwargs.get('price_variable', 'close')
        
        if price_var not in data.columns:
            raise ValueError(f"Price variable {price_var} not found")
        
        prices = data[price_var].dropna()
        returns = prices.pct_change().dropna()
        
        # Simple regime detection based on volatility
        volatility = returns.rolling(20).std()
        high_vol_threshold = volatility.quantile(0.7)
        low_vol_threshold = volatility.quantile(0.3)
        
        # Create regime indicators
        high_vol_regime = volatility > high_vol_threshold
        low_vol_regime = volatility < low_vol_threshold
        
        # Test for regime differences
        high_vol_returns = returns[high_vol_regime]
        low_vol_returns = returns[low_vol_regime]
        
        if len(high_vol_returns) > 0 and len(low_vol_returns) > 0:
            test_statistic, p_value = stats.ttest_ind(high_vol_returns, low_vol_returns)
            effect_size = abs(high_vol_returns.mean() - low_vol_returns.mean()) / np.sqrt(
                (high_vol_returns.var() + low_vol_returns.var()) / 2
            )
        else:
            test_statistic = 0
            p_value = 1.0
            effect_size = 0
        
        is_significant = p_value < self.significance_level
        
        interpretation = f"T-test between regimes: statistic={test_statistic:.3f}, p-value={p_value:.3f}"
        
        recommendations = self._generate_regime_recommendations(
            test_statistic, p_value, effect_size, is_significant
        )
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=(0, 0),  # Not applicable for regime test
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'n_high_vol': len(high_vol_returns),
                'n_low_vol': len(low_vol_returns),
                'high_vol_mean': high_vol_returns.mean() if len(high_vol_returns) > 0 else 0,
                'low_vol_mean': low_vol_returns.mean() if len(low_vol_returns) > 0 else 0,
                'high_vol_std': high_vol_returns.std() if len(high_vol_returns) > 0 else 0,
                'low_vol_std': low_vol_returns.std() if len(low_vol_returns) > 0 else 0
            }
        )
    
    def _test_correlation_hypothesis(self, hypothesis: ResearchHypothesis, 
                                   data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test correlation hypothesis"""
        var1 = kwargs.get('variable1', 'close')
        var2 = kwargs.get('variable2', 'volume')
        
        if var1 not in data.columns or var2 not in data.columns:
            raise ValueError(f"Required variables not found in data")
        
        x = data[var1].dropna()
        y = data[var2].dropna()
        
        # Align data
        common_index = x.index.intersection(y.index)
        x = x.loc[common_index]
        y = y.loc[common_index]
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(x, y)
        test_statistic = correlation
        effect_size = abs(correlation)
        
        # Calculate confidence interval for correlation
        n = len(x)
        if n > 3:
            z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            se = 1 / np.sqrt(n - 3)
            z_ci = stats.norm.interval(self.confidence_level, loc=z, scale=se)
            ci = (np.tanh(z_ci[0]), np.tanh(z_ci[1]))
        else:
            ci = (0, 0)
        
        is_significant = p_value < self.significance_level
        
        interpretation = f"Pearson correlation: {correlation:.3f}, p-value: {p_value:.3f}"
        
        recommendations = self._generate_correlation_recommendations(
            correlation, p_value, effect_size, is_significant
        )
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'n_observations': n,
                'variable1': var1,
                'variable2': var2,
                'var1_mean': x.mean(),
                'var1_std': x.std(),
                'var2_mean': y.mean(),
                'var2_std': y.std()
            }
        )
    
    def _test_anomaly_hypothesis(self, hypothesis: ResearchHypothesis, 
                               data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test anomaly detection hypothesis"""
        price_var = kwargs.get('price_variable', 'close')
        
        if price_var not in data.columns:
            raise ValueError(f"Price variable {price_var} not found")
        
        prices = data[price_var].dropna()
        returns = prices.pct_change().dropna()
        
        # Simple anomaly detection using z-score
        z_scores = np.abs(stats.zscore(returns))
        anomaly_threshold = 3.0
        anomalies = z_scores > anomaly_threshold
        
        # Calculate anomaly rate
        anomaly_rate = anomalies.sum() / len(anomalies)
        
        # Test if anomaly rate is significantly different from expected
        expected_rate = 2 * (1 - stats.norm.cdf(anomaly_threshold))  # Two-tailed
        n = len(anomalies)
        p_value = stats.binom_test(anomalies.sum(), n, expected_rate)
        
        test_statistic = anomaly_rate
        effect_size = abs(anomaly_rate - expected_rate)
        
        is_significant = p_value < self.significance_level
        
        interpretation = f"Anomaly rate: {anomaly_rate:.3f}, expected: {expected_rate:.3f}, p-value: {p_value:.3f}"
        
        recommendations = self._generate_anomaly_recommendations(
            anomaly_rate, expected_rate, p_value, is_significant
        )
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=(0, 0),  # Not applicable for anomaly test
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'n_observations': n,
                'n_anomalies': anomalies.sum(),
                'anomaly_rate': anomaly_rate,
                'expected_rate': expected_rate,
                'z_score_threshold': anomaly_threshold
            }
        )
    
    def _test_backtesting_hypothesis(self, hypothesis: ResearchHypothesis, 
                                   data: pd.DataFrame, **kwargs) -> ResearchResult:
        """Test backtesting hypothesis"""
        # This would integrate with the advanced backtesting engine
        strategy_func = kwargs.get('strategy_function')
        if not strategy_func:
            raise ValueError("Strategy function required for backtesting hypothesis")
        
        # Run backtest (simplified)
        # In practice, would use the AdvancedBacktestingEngine
        returns = data['close'].pct_change().dropna()
        
        # Simple strategy simulation
        signals = strategy_func(data)
        strategy_returns = returns * signals.shift(1).fillna(0)
        
        # Calculate performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (1 + strategy_returns).cumprod().expanding().max()
        drawdown = (1 + strategy_returns).cumprod() / max_drawdown - 1
        max_dd = drawdown.min()
        
        # Test if strategy is significantly better than buy-and-hold
        buy_hold_return = (1 + returns).prod() - 1
        excess_return = total_return - buy_hold_return
        
        # Simple t-test
        test_statistic, p_value = stats.ttest_1samp(strategy_returns, returns.mean())
        effect_size = excess_return
        
        is_significant = p_value < self.significance_level and excess_return > 0
        
        interpretation = f"Strategy return: {total_return:.3f}, Buy-hold: {buy_hold_return:.3f}, p-value: {p_value:.3f}"
        
        recommendations = self._generate_backtesting_recommendations(
            total_return, buy_hold_return, sharpe_ratio, max_dd, p_value, is_significant
        )
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            method=hypothesis.methodology,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=(0, 0),  # Not applicable for backtesting
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations,
            metadata={
                'strategy_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': excess_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'n_trades': (signals != 0).sum()
            }
        )
    
    def _generate_statistical_recommendations(self, test_statistic: float, p_value: float, 
                                            effect_size: float, is_significant: bool) -> List[str]:
        """Generate recommendations for statistical tests"""
        recommendations = []
        
        if is_significant:
            recommendations.append("Hypothesis is statistically significant")
            if effect_size > 0.5:
                recommendations.append("Large effect size detected - strong practical significance")
            elif effect_size > 0.3:
                recommendations.append("Medium effect size detected - moderate practical significance")
            else:
                recommendations.append("Small effect size detected - weak practical significance")
        else:
            recommendations.append("Hypothesis is not statistically significant")
            recommendations.append("Consider increasing sample size or checking data quality")
        
        if p_value < 0.001:
            recommendations.append("Very strong evidence against null hypothesis")
        elif p_value < 0.01:
            recommendations.append("Strong evidence against null hypothesis")
        elif p_value < 0.05:
            recommendations.append("Moderate evidence against null hypothesis")
        
        return recommendations
    
    def _generate_ml_recommendations(self, r2: float, f_statistic: float, 
                                   p_value: float, is_significant: bool) -> List[str]:
        """Generate recommendations for ML tests"""
        recommendations = []
        
        if is_significant:
            recommendations.append("Model is statistically significant")
            if r2 > 0.7:
                recommendations.append("High R² - model explains most variance")
            elif r2 > 0.5:
                recommendations.append("Moderate R² - model explains substantial variance")
            elif r2 > 0.3:
                recommendations.append("Low R² - model explains some variance")
            else:
                recommendations.append("Very low R² - model explains little variance")
        else:
            recommendations.append("Model is not statistically significant")
            recommendations.append("Consider feature engineering or different algorithms")
        
        if f_statistic > 10:
            recommendations.append("High F-statistic - strong model fit")
        elif f_statistic > 5:
            recommendations.append("Moderate F-statistic - reasonable model fit")
        else:
            recommendations.append("Low F-statistic - weak model fit")
        
        return recommendations
    
    def _generate_timeseries_recommendations(self, test_statistic: float, p_value: float, 
                                           effect_size: float, is_significant: bool) -> List[str]:
        """Generate recommendations for time series tests"""
        recommendations = []
        
        if is_significant:
            recommendations.append("Significant autocorrelation detected")
            recommendations.append("Consider ARIMA or other time series models")
            recommendations.append("Data may not be suitable for standard statistical tests")
        else:
            recommendations.append("No significant autocorrelation detected")
            recommendations.append("Data appears to be white noise")
            recommendations.append("Standard statistical tests are appropriate")
        
        if effect_size > 0.3:
            recommendations.append("Strong autocorrelation - high predictability")
        elif effect_size > 0.1:
            recommendations.append("Moderate autocorrelation - some predictability")
        else:
            recommendations.append("Weak autocorrelation - low predictability")
        
        return recommendations
    
    def _generate_regime_recommendations(self, test_statistic: float, p_value: float, 
                                       effect_size: float, is_significant: bool) -> List[str]:
        """Generate recommendations for regime tests"""
        recommendations = []
        
        if is_significant:
            recommendations.append("Significant regime differences detected")
            recommendations.append("Consider regime-switching models")
            recommendations.append("Different strategies may be needed for different regimes")
        else:
            recommendations.append("No significant regime differences detected")
            recommendations.append("Single strategy may work across all periods")
        
        if effect_size > 0.5:
            recommendations.append("Large regime effect - strong differentiation needed")
        elif effect_size > 0.3:
            recommendations.append("Moderate regime effect - some differentiation needed")
        else:
            recommendations.append("Small regime effect - minimal differentiation needed")
        
        return recommendations
    
    def _generate_correlation_recommendations(self, correlation: float, p_value: float, 
                                            effect_size: float, is_significant: bool) -> List[str]:
        """Generate recommendations for correlation tests"""
        recommendations = []
        
        if is_significant:
            recommendations.append("Significant correlation detected")
            if abs(correlation) > 0.7:
                recommendations.append("Strong correlation - high linear relationship")
            elif abs(correlation) > 0.5:
                recommendations.append("Moderate correlation - moderate linear relationship")
            elif abs(correlation) > 0.3:
                recommendations.append("Weak correlation - low linear relationship")
        else:
            recommendations.append("No significant correlation detected")
            recommendations.append("Variables appear to be independent")
        
        if correlation > 0:
            recommendations.append("Positive correlation - variables move together")
        else:
            recommendations.append("Negative correlation - variables move oppositely")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, anomaly_rate: float, expected_rate: float, 
                                        p_value: float, is_significant: bool) -> List[str]:
        """Generate recommendations for anomaly tests"""
        recommendations = []
        
        if is_significant:
            if anomaly_rate > expected_rate:
                recommendations.append("More anomalies than expected - high volatility period")
                recommendations.append("Consider risk management adjustments")
            else:
                recommendations.append("Fewer anomalies than expected - low volatility period")
                recommendations.append("Consider increasing position sizes")
        else:
            recommendations.append("Anomaly rate is within expected range")
            recommendations.append("Normal market conditions")
        
        if anomaly_rate > 0.1:
            recommendations.append("High anomaly rate - consider robust models")
        elif anomaly_rate < 0.01:
            recommendations.append("Low anomaly rate - standard models should work")
        
        return recommendations
    
    def _generate_backtesting_recommendations(self, strategy_return: float, buy_hold_return: float, 
                                            sharpe_ratio: float, max_drawdown: float, 
                                            p_value: float, is_significant: bool) -> List[str]:
        """Generate recommendations for backtesting tests"""
        recommendations = []
        
        if is_significant:
            recommendations.append("Strategy significantly outperforms buy-and-hold")
            if strategy_return > buy_hold_return * 1.5:
                recommendations.append("Strong outperformance - consider increasing allocation")
            elif strategy_return > buy_hold_return * 1.2:
                recommendations.append("Moderate outperformance - good strategy")
            else:
                recommendations.append("Weak outperformance - consider optimization")
        else:
            recommendations.append("Strategy does not significantly outperform buy-and-hold")
            recommendations.append("Consider strategy revision or different approach")
        
        if sharpe_ratio > 1.5:
            recommendations.append("Excellent risk-adjusted returns")
        elif sharpe_ratio > 1.0:
            recommendations.append("Good risk-adjusted returns")
        elif sharpe_ratio > 0.5:
            recommendations.append("Moderate risk-adjusted returns")
        else:
            recommendations.append("Poor risk-adjusted returns")
        
        if abs(max_drawdown) > 0.2:
            recommendations.append("High drawdown risk - consider position sizing")
        elif abs(max_drawdown) > 0.1:
            recommendations.append("Moderate drawdown risk - monitor closely")
        else:
            recommendations.append("Low drawdown risk - good risk management")
        
        return recommendations
    
    def _log_research_activity(self, hypothesis: ResearchHypothesis, result: ResearchResult):
        """Log research activity"""
        activity = {
            'timestamp': datetime.now(),
            'hypothesis_id': hypothesis.id,
            'hypothesis_title': hypothesis.title,
            'method': result.method.value,
            'is_significant': result.is_significant,
            'p_value': result.p_value,
            'effect_size': result.effect_size
        }
        self.research_log.append(activity)
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive research summary"""
        total_hypotheses = len(self.hypotheses)
        completed_hypotheses = len([h for h in self.hypotheses.values() if h.status == "completed"])
        significant_results = len([r for r in self.results if r.is_significant])
        
        # Method breakdown
        method_counts = {}
        for result in self.results:
            method = result.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Success rate by method
        method_success = {}
        for method in method_counts.keys():
            method_results = [r for r in self.results if r.method.value == method]
            success_rate = len([r for r in method_results if r.is_significant]) / len(method_results)
            method_success[method] = success_rate
        
        return {
            'total_hypotheses': total_hypotheses,
            'completed_hypotheses': completed_hypotheses,
            'significant_results': significant_results,
            'overall_success_rate': significant_results / len(self.results) if self.results else 0,
            'method_breakdown': method_counts,
            'method_success_rates': method_success,
            'research_log_entries': len(self.research_log)
        }
    
    def export_results(self, filename: str) -> bool:
        """Export research results to JSON file"""
        try:
            export_data = {
                'hypotheses': {
                    hid: {
                        'id': h.id,
                        'title': h.title,
                        'description': h.description,
                        'methodology': h.methodology.value,
                        'expected_outcome': h.expected_outcome,
                        'confidence_level': h.confidence_level,
                        'status': h.status,
                        'results': h.results
                    } for hid, h in self.hypotheses.items()
                },
                'results': [
                    {
                        'hypothesis_id': r.hypothesis_id,
                        'method': r.method.value,
                        'test_statistic': r.test_statistic,
                        'p_value': r.p_value,
                        'confidence_interval': r.confidence_interval,
                        'effect_size': r.effect_size,
                        'is_significant': r.is_significant,
                        'interpretation': r.interpretation,
                        'recommendations': r.recommendations,
                        'metadata': r.metadata
                    } for r in self.results
                ],
                'summary': self.get_research_summary()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize research system
    research = ResearchMethodologySystem()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.randn(252) * 0.01)
    volumes = np.random.randint(1000, 10000, 252)
    
    data = pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'returns': pd.Series(prices).pct_change()
    }, index=dates)
    
    # Create hypotheses
    hyp1 = research.create_hypothesis(
        "Price-Volume Correlation",
        "Test if there is a significant correlation between price and volume",
        ResearchMethod.CORRELATION_ANALYSIS,
        "Positive correlation expected"
    )
    
    hyp2 = research.create_hypothesis(
        "Returns Normality",
        "Test if returns are normally distributed",
        ResearchMethod.STATISTICAL_ANALYSIS,
        "Returns should be approximately normal"
    )
    
    # Test hypotheses
    result1 = research.test_hypothesis(hyp1, data, variable1='close', variable2='volume')
    result2 = research.test_hypothesis(hyp2, data, x_variable='returns')
    
    # Get summary
    summary = research.get_research_summary()
    print("Research Summary:")
    print(f"Total hypotheses: {summary['total_hypotheses']}")
    print(f"Significant results: {summary['significant_results']}")
    print(f"Success rate: {summary['overall_success_rate']:.2%}")
    
    # Export results
    research.export_results("research_results.json")
    print("Results exported to research_results.json")
