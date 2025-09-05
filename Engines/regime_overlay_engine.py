#!/usr/bin/env python3
"""
Regime Overlay Engine

This engine integrates historical regime analysis with backtesting engines,
providing intelligent decision-making capabilities based on market regimes.

Features:
- Load historical regime data
- Overlay regime information on backtesting results
- Provide regime-aware strategy recommendations
- Generate regime-based performance analysis
- Create regime transition alerts
- Integrate with bot decision-making systems
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')

# Import base engine
from .core_engine import CoreEngine, EngineConfig

@dataclass
class RegimeOverlayConfig(EngineConfig):
    """Configuration for regime overlay engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    strategies_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    
    # Regime data paths
    regime_data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Results"
    regime_file_pattern: str = "regime_analysis_detailed_*.json"
    
    # Regime overlay settings
    enable_regime_filtering: bool = True
    regime_confidence_threshold: float = 0.7
    regime_stability_threshold: float = 0.6
    
    # Strategy recommendations
    enable_strategy_recommendations: bool = True
    regime_strategy_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        'Bull': ['momentum', 'trend_following', 'breakout'],
        'Bear': ['mean_reversion', 'short_selling', 'defensive'],
        'Sideways': ['mean_reversion', 'range_trading', 'scalping'],
        'High_Volatility': ['volatility_trading', 'options_strategies'],
        'Low_Volatility': ['trend_following', 'momentum']
    })
    
    # Performance analysis
    enable_regime_performance_analysis: bool = True
    regime_performance_metrics: List[str] = field(default_factory=lambda: [
        'regime_return', 'regime_sharpe', 'regime_drawdown', 'regime_win_rate'
    ])
    
    # Alerts and notifications
    enable_regime_alerts: bool = True
    alert_regime_transitions: bool = True
    alert_performance_degradation: bool = True

@dataclass
class RegimeOverlayResult:
    """Result of regime overlay analysis"""
    data_file: str
    strategy_name: str
    regime_analysis: Dict[str, Any]
    regime_recommendations: List[str]
    regime_performance: Dict[str, float]
    regime_alerts: List[Dict[str, Any]]
    overlay_timestamp: datetime = field(default_factory=datetime.now)

class RegimeOverlayEngine(CoreEngine):
    """Engine for overlaying regime analysis on backtesting results"""
    
    def __init__(self, config: RegimeOverlayConfig):
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.regime_data = {}
        self.regime_metadata = {}
    
    def load_regime_data(self) -> bool:
        """Load historical regime analysis data"""
        self.logger.info("Loading regime analysis data...")
        
        regime_path = Path(self.config.regime_data_path)
        if not regime_path.exists():
            self.logger.warning(f"Regime data path does not exist: {regime_path}")
            return False
        
        # Find regime analysis files
        regime_files = list(regime_path.glob(self.config.regime_file_pattern))
        if not regime_files:
            self.logger.warning("No regime analysis files found")
            return False
        
        # Load the most recent file
        latest_file = max(regime_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                regime_data = json.load(f)
            
            # Process regime data
            for item in regime_data:
                data_file = item['data_file']
                self.regime_data[data_file] = item
                self.regime_metadata[data_file] = {
                    'token': item.get('token', 'Unknown'),
                    'timeframe': item.get('timeframe', 'Unknown'),
                    'regimes_detected': item.get('regimes_detected', 0),
                    'regime_stability': item.get('regime_stability', 0.0)
                }
            
            self.logger.info(f"Loaded regime data for {len(self.regime_data)} files")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading regime data: {e}")
            return False
    
    def get_regime_for_period(self, data_file: str, timestamp: datetime) -> Optional[str]:
        """Get regime for a specific period"""
        if data_file not in self.regime_data:
            return None
        
        regime_data = self.regime_data[data_file]
        regime_transitions = regime_data.get('regime_transitions', [])
        
        # Find the regime for the given timestamp
        current_regime = None
        for transition in regime_transitions:
            transition_time = datetime.fromisoformat(transition['timestamp'])
            if timestamp >= transition_time:
                current_regime = transition['to_regime']
            else:
                break
        
        return current_regime
    
    def analyze_regime_performance(self, backtest_results: List[Dict[str, Any]], 
                                 data_file: str) -> Dict[str, float]:
        """Analyze performance by regime"""
        if data_file not in self.regime_data:
            return {}
        
        regime_data = self.regime_data[data_file]
        regime_stats = regime_data.get('regime_statistics', {})
        
        # Calculate regime-weighted performance metrics
        regime_performance = {}
        
        for regime, stats in regime_stats.items():
            regime_performance[f"{regime}_return"] = stats.get('mean_return', 0.0)
            regime_performance[f"{regime}_sharpe"] = stats.get('sharpe_ratio', 0.0)
            regime_performance[f"{regime}_win_rate"] = stats.get('win_rate', 0.0)
            regime_performance[f"{regime}_volatility"] = stats.get('std_return', 0.0)
        
        return regime_performance
    
    def generate_regime_recommendations(self, data_file: str, 
                                      current_regime: Optional[str] = None) -> List[str]:
        """Generate strategy recommendations based on regime"""
        if not self.config.enable_strategy_recommendations:
            return []
        
        recommendations = []
        
        # Get regime-specific recommendations
        if current_regime and current_regime in self.config.regime_strategy_mapping:
            recommendations.extend(self.config.regime_strategy_mapping[current_regime])
        
        # Add general recommendations based on regime stability
        if data_file in self.regime_metadata:
            stability = self.regime_metadata[data_file]['regime_stability']
            
            if stability < self.config.regime_stability_threshold:
                recommendations.append('conservative_position_sizing')
                recommendations.append('frequent_rebalancing')
            else:
                recommendations.append('standard_position_sizing')
                recommendations.append('periodic_rebalancing')
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_regime_alerts(self, data_file: str, backtest_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate regime-based alerts"""
        alerts = []
        
        if not self.config.enable_regime_alerts:
            return alerts
        
        if data_file not in self.regime_data:
            return alerts
        
        regime_data = self.regime_data[data_file]
        
        # Check for regime transitions
        if self.config.alert_regime_transitions:
            transitions = regime_data.get('regime_transitions', [])
            recent_transitions = [
                t for t in transitions 
                if (datetime.now() - datetime.fromisoformat(t['timestamp'])).days <= 7
            ]
            
            for transition in recent_transitions:
                alerts.append({
                    'type': 'regime_transition',
                    'severity': 'medium',
                    'message': f"Recent regime transition: {transition['from_regime']} to {transition['to_regime']}",
                    'timestamp': transition['timestamp'],
                    'recommendation': 'Review strategy parameters and position sizing'
                })
        
        # Check for performance degradation
        if self.config.alert_performance_degradation:
            regime_stats = regime_data.get('regime_statistics', {})
            
            for regime, stats in regime_stats.items():
                win_rate = stats.get('win_rate', 0)
                sharpe_ratio = stats.get('sharpe_ratio', 0)
                
                if win_rate < 40 or sharpe_ratio < 0.5:
                    alerts.append({
                        'type': 'performance_degradation',
                        'severity': 'high',
                        'message': f"Poor performance in {regime} regime: Win rate {win_rate:.1f}%, Sharpe {sharpe_ratio:.2f}",
                        'regime': regime,
                        'recommendation': 'Consider alternative strategies for this regime'
                    })
        
        return alerts
    
    def overlay_regime_analysis(self, backtest_results: List[Dict[str, Any]], 
                              data_file: str) -> RegimeOverlayResult:
        """Overlay regime analysis on backtest results"""
        self.logger.info(f"Overlaying regime analysis for {data_file}")
        
        # Analyze regime performance
        regime_performance = self.analyze_regime_performance(backtest_results, data_file)
        
        # Get current regime (use latest timestamp from backtest)
        current_regime = None
        if backtest_results:
            latest_result = max(backtest_results, key=lambda x: x.get('timestamp', datetime.min))
            if 'timestamp' in latest_result:
                current_regime = self.get_regime_for_period(data_file, latest_result['timestamp'])
        
        # Generate recommendations
        recommendations = self.generate_regime_recommendations(data_file, current_regime)
        
        # Generate alerts
        alerts = self.generate_regime_alerts(data_file, backtest_results)
        
        # Create overlay result
        overlay_result = RegimeOverlayResult(
            data_file=data_file,
            strategy_name=backtest_results[0].get('strategy_name', 'Unknown') if backtest_results else 'Unknown',
            regime_analysis={
                'current_regime': current_regime,
                'regime_stability': self.regime_metadata.get(data_file, {}).get('regime_stability', 0.0),
                'regimes_detected': self.regime_metadata.get(data_file, {}).get('regimes_detected', 0)
            },
            regime_recommendations=recommendations,
            regime_performance=regime_performance,
            regime_alerts=alerts
        )
        
        return overlay_result
    
    def create_regime_aware_strategy_config(self, data_file: str, 
                                          base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create regime-aware strategy configuration"""
        if data_file not in self.regime_data:
            return base_config
        
        regime_data = self.regime_data[data_file]
        regime_stats = regime_data.get('regime_statistics', {})
        
        # Adjust configuration based on regime characteristics
        modified_config = base_config.copy()
        
        # Adjust position sizing based on regime volatility
        avg_volatility = np.mean([stats.get('std_return', 0.02) for stats in regime_stats.values()])
        
        if avg_volatility > 0.03:  # High volatility
            modified_config['position_size_multiplier'] = 0.5
            modified_config['stop_loss_multiplier'] = 1.5
        elif avg_volatility < 0.01:  # Low volatility
            modified_config['position_size_multiplier'] = 1.2
            modified_config['stop_loss_multiplier'] = 0.8
        
        # Adjust rebalancing frequency based on regime stability
        stability = self.regime_metadata.get(data_file, {}).get('regime_stability', 0.5)
        
        if stability < 0.5:  # Unstable regimes
            modified_config['rebalancing_frequency'] = 'daily'
        else:  # Stable regimes
            modified_config['rebalancing_frequency'] = 'weekly'
        
        return modified_config
    
    def export_regime_intelligence(self, overlay_results: List[RegimeOverlayResult]) -> str:
        """Export regime intelligence for bot integration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.config.results_path}/regime_intelligence_{timestamp}"
        
        os.makedirs(output_path, exist_ok=True)
        
        # Export overlay results
        overlay_data = []
        for result in overlay_results:
            overlay_data.append({
                'data_file': result.data_file,
                'strategy_name': result.strategy_name,
                'regime_analysis': result.regime_analysis,
                'regime_recommendations': result.regime_recommendations,
                'regime_performance': result.regime_performance,
                'regime_alerts': result.regime_alerts,
                'overlay_timestamp': result.overlay_timestamp.isoformat()
            })
        
        # Save as JSON
        json_path = f"{output_path}/regime_intelligence.json"
        with open(json_path, 'w') as f:
            json.dump(overlay_data, f, indent=2)
        
        # Create bot integration file
        bot_integration_path = f"{output_path}/bot_integration.py"
        self.create_bot_integration_file(bot_integration_path, overlay_data)
        
        self.logger.info(f"Regime intelligence exported to: {output_path}")
        return output_path
    
    def create_bot_integration_file(self, filepath: str, overlay_data: List[Dict[str, Any]]):
        """Create Python file for bot integration"""
        bot_code = '''#!/usr/bin/env python3
"""
Regime Intelligence Integration for Trading Bots

This file provides regime-aware decision making capabilities for trading bots.
Generated automatically by the Regime Overlay Engine.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class RegimeIntelligence:
    """Regime intelligence for bot decision making"""
    
    def __init__(self, regime_data_path: str):
        self.regime_data = self.load_regime_data(regime_data_path)
        self.current_regimes = {}
    
    def load_regime_data(self, filepath: str) -> Dict[str, Any]:
        """Load regime intelligence data"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading regime data: {e}")
            return {}
    
    def get_regime_recommendations(self, data_file: str) -> List[str]:
        """Get regime-based strategy recommendations"""
        for item in self.regime_data:
            if item['data_file'] == data_file:
                return item.get('regime_recommendations', [])
        return []
    
    def get_regime_alerts(self, data_file: str) -> List[Dict[str, Any]]:
        """Get regime-based alerts"""
        for item in self.regime_data:
            if item['data_file'] == data_file:
                return item.get('regime_alerts', [])
        return []
    
    def should_adjust_position_size(self, data_file: str, base_size: float) -> float:
        """Adjust position size based on regime"""
        for item in self.regime_data:
            if item['data_file'] == data_file:
                regime_analysis = item.get('regime_analysis', {})
                stability = regime_analysis.get('regime_stability', 0.5)
                
                # Reduce position size in unstable regimes
                if stability < 0.5:
                    return base_size * 0.7
                elif stability > 0.8:
                    return base_size * 1.1
                
        return base_size
    
    def should_pause_trading(self, data_file: str) -> bool:
        """Determine if trading should be paused based on regime"""
        alerts = self.get_regime_alerts(data_file)
        
        for alert in alerts:
            if alert.get('severity') == 'high' and alert.get('type') == 'performance_degradation':
                return True
        
        return False
    
    def get_optimal_strategy(self, data_file: str, available_strategies: List[str]) -> Optional[str]:
        """Get optimal strategy based on current regime"""
        recommendations = self.get_regime_recommendations(data_file)
        
        for recommendation in recommendations:
            if recommendation in available_strategies:
                return recommendation
        
        return None

# Example usage
if __name__ == "__main__":
    # Initialize regime intelligence
    regime_intel = RegimeIntelligence("regime_intelligence.json")
    
    # Example bot decision making
    data_file = "BTCUSDT_1h.csv"
    
    # Check if trading should be paused
    if regime_intel.should_pause_trading(data_file):
        print("Trading paused due to regime conditions")
    else:
        # Get optimal strategy
        available_strategies = ['momentum', 'mean_reversion', 'trend_following']
        optimal_strategy = regime_intel.get_optimal_strategy(data_file, available_strategies)
        
        if optimal_strategy:
            print(f"Recommended strategy: {optimal_strategy}")
        
        # Adjust position size
        base_position_size = 0.1
        adjusted_size = regime_intel.should_adjust_position_size(data_file, base_position_size)
        print(f"Adjusted position size: {adjusted_size}")
'''
        
        with open(filepath, 'w') as f:
            f.write(bot_code)
    
    async def run_regime_overlay_analysis(self, backtest_results: List[Dict[str, Any]]) -> List[RegimeOverlayResult]:
        """Run complete regime overlay analysis"""
        self.logger.info("Starting regime overlay analysis...")
        
        # Load regime data
        if not self.load_regime_data():
            self.logger.error("Failed to load regime data")
            return []
        
        # Group results by data file
        results_by_file = {}
        for result in backtest_results:
            data_file = result.get('data_file', 'Unknown')
            if data_file not in results_by_file:
                results_by_file[data_file] = []
            results_by_file[data_file].append(result)
        
        # Overlay regime analysis for each file
        overlay_results = []
        for data_file, file_results in results_by_file.items():
            overlay_result = self.overlay_regime_analysis(file_results, data_file)
            overlay_results.append(overlay_result)
        
        # Export regime intelligence
        if overlay_results:
            self.export_regime_intelligence(overlay_results)
        
        self.logger.info(f"Regime overlay analysis complete. Processed {len(overlay_results)} files")
        return overlay_results

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = RegimeOverlayConfig(
        data_path="./Data",
        results_path="./Results/RegimeOverlay",
        regime_data_path="./Results/RegimeAnalysis",
        enable_regime_filtering=True,
        enable_strategy_recommendations=True,
        enable_regime_alerts=True
    )
    
    # Create engine
    engine = RegimeOverlayEngine(config)
    
    print("Regime Overlay Engine created successfully!")
    print("Ready to provide regime-aware decision making capabilities.")
