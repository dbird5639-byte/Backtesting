#!/usr/bin/env python3
"""
Comprehensive Regime Analysis and Backtesting Script

This script orchestrates the complete regime analysis and backtesting workflow:
1. Analyzes historical data to identify market regimes
2. Runs backtesting with regime-aware strategies
3. Generates comprehensive visualizations with regime overlays
4. Provides regime intelligence for bot decision-making

Features:
- Historical regime analysis across all data files
- Regime-aware backtesting with multiple engines
- Comprehensive visualization with regime overlays
- Bot integration files for intelligent decision-making
- Organized results by strategy and data file
"""

import os
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import engines
from Engines.core_engine import CoreEngine, EngineConfig
from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
from Engines.engine_factory import EngineFactory, EngineFactoryConfig, EngineType, ExecutionMode

# Import regime analysis
from scripts.regime_analysis import RegimeAnalyzer, RegimeConfig

class ComprehensiveRegimeAnalysis:
    """Orchestrates comprehensive regime analysis and backtesting"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        
        # Initialize configurations
        self.regime_config = RegimeConfig(
            data_path="./Data",
            results_path="./Results/RegimeAnalysis",
            save_csv=True,
            save_json=True,
            save_plots=True,
            save_heatmaps=True
        )
        
        self.core_config = EngineConfig(
            data_path="./Data",
            results_path="./Results/CoreEngine",
            save_csv=True,
            save_json=False,
            save_plots=False
        )
        
        self.visualization_config = VisualizationConfig(
            data_path="./Data",
            results_path="./Results/Visualizations",
            save_csv=True,
            save_json=True,
            save_png=True,
            save_html=True,
            enable_regime_overlay=True,
            organize_by_strategy=True,
            organize_by_data_file=True,
            create_summary_dashboard=True
        )
        
        self.regime_overlay_config = RegimeOverlayConfig(
            data_path="./Data",
            results_path="./Results/RegimeOverlay",
            regime_data_path="./Results/RegimeAnalysis",
            enable_regime_filtering=True,
            enable_strategy_recommendations=True,
            enable_regime_alerts=True
        )
        
        # Initialize engines
        self.regime_analyzer = RegimeAnalyzer(self.regime_config)
        self.core_engine = CoreEngine(self.core_config)
        self.visualization_engine = EnhancedVisualizationEngine(self.visualization_config)
        self.regime_overlay_engine = RegimeOverlayEngine(self.regime_overlay_config)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'comprehensive_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def discover_data_files(self) -> List[str]:
        """Discover all data files"""
        data_path = Path(self.regime_config.data_path)
        if not data_path.exists():
            self.logger.warning(f"Data path does not exist: {data_path}")
            return []
        
        csv_files = list(data_path.rglob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files")
        return [str(f) for f in csv_files]
    
    def run_regime_analysis(self) -> List[Any]:
        """Run historical regime analysis"""
        self.logger.info("🔍 Starting historical regime analysis...")
        
        try:
            results = self.regime_analyzer.run_analysis()
            self.logger.info(f"✅ Regime analysis complete. Processed {len(results)} files")
            return results
        except Exception as e:
            self.logger.error(f"❌ Regime analysis failed: {e}")
            return []
    
    async def run_core_backtesting(self, data_files: List[str]) -> List[Dict[str, Any]]:
        """Run core backtesting on all data files"""
        self.logger.info("🔧 Starting core backtesting...")
        
        try:
            # Load strategies
            strategies = self.load_strategies()
            if not strategies:
                self.logger.warning("No strategies loaded")
                return []
            
            results = []
            for data_file in data_files:
                self.logger.info(f"Processing {data_file}")
                
                # Load data
                data = pd.read_csv(data_file)
                if len(data) < 100:
                    self.logger.warning(f"Insufficient data in {data_file}")
                    continue
                
                # Run backtesting for each strategy
                for strategy_name, strategy_config in strategies.items():
                    try:
                        # Create strategy instance
                        strategy = self.create_strategy_instance(strategy_name, strategy_config)
                        
                        # Run backtest
                        result = await self.core_engine.run_backtest(data, strategy)
                        
                        # Add metadata
                        result['data_file'] = data_file
                        result['strategy_name'] = strategy_name
                        result['data'] = data  # Include data for visualization
                        
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Error running {strategy_name} on {data_file}: {e}")
            
            self.logger.info(f"✅ Core backtesting complete. Generated {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Core backtesting failed: {e}")
            return []
    
    def load_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load available strategies"""
        strategies = {
            'SimpleMA': {'short_window': 10, 'long_window': 20},
            'AdvancedMA': {'short_window': 15, 'long_window': 30},
            'Momentum': {'lookback_period': 20, 'threshold': 0.02},
            'MeanReversion': {'lookback_period': 20, 'threshold': 0.02},
            'VolatilityBreakout': {'lookback_period': 20, 'multiplier': 2.0}
        }
        
        self.logger.info(f"Loaded {len(strategies)} strategies")
        return strategies
    
    def create_strategy_instance(self, strategy_name: str, config: Dict[str, Any]):
        """Create strategy instance (placeholder - implement based on your strategy classes)"""
        # This is a placeholder - you'll need to implement based on your actual strategy classes
        class SimpleStrategy:
            def __init__(self, name, config):
                self.name = name
                self.config = config
            
            def generate_signals(self, data):
                # Simple moving average crossover strategy
                short_window = self.config.get('short_window', 10)
                long_window = self.config.get('long_window', 20)
                
                data['short_ma'] = data['close'].rolling(window=short_window).mean()
                data['long_ma'] = data['close'].rolling(window=long_window).mean()
                
                data['signal'] = 0
                data['signal'][short_window:] = np.where(
                    data['short_ma'][short_window:] > data['long_ma'][short_window:], 1, 0
                )
                data['positions'] = data['signal'].diff()
                
                return data
        
        return SimpleStrategy(strategy_name, config)
    
    async def run_enhanced_visualization(self, backtest_results: List[Dict[str, Any]], 
                                       regime_data: Optional[Dict[str, pd.DataFrame]] = None) -> str:
        """Run enhanced visualization with regime overlays"""
        self.logger.info("📊 Starting enhanced visualization...")
        
        try:
            visualization_path = await self.visualization_engine.run_visualization_analysis(
                backtest_results, regime_data
            )
            self.logger.info(f"✅ Visualization complete. Results saved to: {visualization_path}")
            return visualization_path
        except Exception as e:
            self.logger.error(f"❌ Visualization failed: {e}")
            return ""
    
    async def run_regime_overlay_analysis(self, backtest_results: List[Dict[str, Any]]) -> List[Any]:
        """Run regime overlay analysis"""
        self.logger.info("🎯 Starting regime overlay analysis...")
        
        try:
            overlay_results = await self.regime_overlay_engine.run_regime_overlay_analysis(backtest_results)
            self.logger.info(f"✅ Regime overlay analysis complete. Generated {len(overlay_results)} overlays")
            return overlay_results
        except Exception as e:
            self.logger.error(f"❌ Regime overlay analysis failed: {e}")
            return []
    
    def create_bot_integration_summary(self, regime_results: List[Any], 
                                     overlay_results: List[Any]) -> str:
        """Create comprehensive bot integration summary"""
        self.logger.info("🤖 Creating bot integration summary...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"./Results/BotIntegration/bot_integration_summary_{timestamp}"
        os.makedirs(summary_path, exist_ok=True)
        
        # Create comprehensive summary
        summary_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'regime_analysis_summary': {
                'total_files_analyzed': len(regime_results),
                'regime_types_detected': self.get_regime_types(regime_results),
                'average_stability': self.calculate_average_stability(regime_results)
            },
            'backtesting_summary': {
                'total_strategies_tested': len(set(r.get('strategy_name', 'Unknown') for r in overlay_results)),
                'total_data_files': len(set(r.data_file for r in overlay_results)),
                'regime_aware_results': len(overlay_results)
            },
            'recommendations': self.generate_bot_recommendations(overlay_results),
            'alerts': self.consolidate_alerts(overlay_results)
        }
        
        # Save summary
        summary_file = f"{summary_path}/comprehensive_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create bot configuration file
        bot_config_file = f"{summary_path}/bot_config.py"
        self.create_bot_config_file(bot_config_file, summary_data)
        
        self.logger.info(f"✅ Bot integration summary created: {summary_path}")
        return summary_path
    
    def get_regime_types(self, regime_results: List[Any]) -> List[str]:
        """Get unique regime types from analysis"""
        regime_types = set()
        for result in regime_results:
            if hasattr(result, 'regime_breakdown'):
                regime_types.update(result.regime_breakdown.keys())
        return list(regime_types)
    
    def calculate_average_stability(self, regime_results: List[Any]) -> float:
        """Calculate average regime stability"""
        if not regime_results:
            return 0.0
        
        stabilities = [r.regime_stability for r in regime_results if hasattr(r, 'regime_stability')]
        return np.mean(stabilities) if stabilities else 0.0
    
    def generate_bot_recommendations(self, overlay_results: List[Any]) -> List[str]:
        """Generate comprehensive bot recommendations"""
        recommendations = set()
        
        for result in overlay_results:
            if hasattr(result, 'regime_recommendations'):
                recommendations.update(result.regime_recommendations)
        
        return list(recommendations)
    
    def consolidate_alerts(self, overlay_results: List[Any]) -> List[Dict[str, Any]]:
        """Consolidate all alerts"""
        all_alerts = []
        
        for result in overlay_results:
            if hasattr(result, 'regime_alerts'):
                all_alerts.extend(result.regime_alerts)
        
        return all_alerts
    
    def create_bot_config_file(self, filepath: str, summary_data: Dict[str, Any]):
        """Create bot configuration file"""
        bot_config = f'''#!/usr/bin/env python3
"""
Comprehensive Bot Configuration
Generated by Comprehensive Regime Analysis

This file provides intelligent configuration for trading bots based on
comprehensive regime analysis and backtesting results.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class IntelligentBotConfig:
    """Intelligent bot configuration based on regime analysis"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
        self.regime_recommendations = config_data.get('recommendations', [])
        self.alerts = config_data.get('alerts', [])
        self.regime_summary = config_data.get('regime_analysis_summary', {{}})
        self.backtesting_summary = config_data.get('backtesting_summary', {{}})
    
    def get_position_sizing_config(self) -> Dict[str, Any]:
        """Get position sizing configuration based on regime analysis"""
        avg_stability = self.regime_summary.get('average_stability', 0.5)
        
        if avg_stability < 0.4:
            return {{
                'base_position_size': 0.05,
                'max_position_size': 0.1,
                'regime_adjustment_factor': 0.5
            }}
        elif avg_stability > 0.7:
            return {{
                'base_position_size': 0.1,
                'max_position_size': 0.2,
                'regime_adjustment_factor': 1.2
            }}
        else:
            return {{
                'base_position_size': 0.08,
                'max_position_size': 0.15,
                'regime_adjustment_factor': 1.0
            }}
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {{
            'max_drawdown': 0.15,
            'stop_loss_multiplier': 1.5,
            'take_profit_multiplier': 2.0,
            'regime_aware_stops': True,
            'dynamic_position_sizing': True
        }}
    
    def get_strategy_selection_config(self) -> Dict[str, Any]:
        """Get strategy selection configuration"""
        return {{
            'preferred_strategies': self.regime_recommendations,
            'strategy_rotation': True,
            'regime_based_selection': True,
            'fallback_strategy': 'conservative'
        }}
    
    def get_alerts_config(self) -> Dict[str, Any]:
        """Get alerts configuration"""
        return {{
            'enable_regime_alerts': True,
            'enable_performance_alerts': True,
            'alert_severity_threshold': 'medium',
            'alert_channels': ['log', 'email']
        }}

# Load configuration
with open('comprehensive_summary.json', 'r') as f:
    config_data = json.load(f)

# Create intelligent bot configuration
bot_config = IntelligentBotConfig(config_data)

# Export configurations
position_config = bot_config.get_position_sizing_config()
risk_config = bot_config.get_risk_management_config()
strategy_config = bot_config.get_strategy_selection_config()
alerts_config = bot_config.get_alerts_config()

print("Intelligent Bot Configuration Generated:")
print(f"Position Sizing: {{position_config}}")
print(f"Risk Management: {{risk_config}}")
print(f"Strategy Selection: {{strategy_config}}")
print(f"Alerts: {{alerts_config}}")
'''
        
        with open(filepath, 'w') as f:
            f.write(bot_config)
    
    async def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis"""
        self.logger.info("🚀 Starting Comprehensive Regime Analysis and Backtesting")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Discover data files
            data_files = self.discover_data_files()
            if not data_files:
                self.logger.error("No data files found. Exiting.")
                return
            
            # Step 2: Run regime analysis
            regime_results = self.run_regime_analysis()
            
            # Step 3: Run core backtesting
            backtest_results = await self.run_core_backtesting(data_files)
            
            # Step 4: Run enhanced visualization
            visualization_path = await self.run_enhanced_visualization(backtest_results)
            
            # Step 5: Run regime overlay analysis
            overlay_results = await self.run_regime_overlay_analysis(backtest_results)
            
            # Step 6: Create bot integration summary
            bot_integration_path = self.create_bot_integration_summary(regime_results, overlay_results)
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Total Duration: {duration}")
            self.logger.info(f"📁 Data Files Processed: {len(data_files)}")
            self.logger.info(f"🔍 Regime Analysis Results: {len(regime_results)}")
            self.logger.info(f"🔧 Backtest Results: {len(backtest_results)}")
            self.logger.info(f"📊 Visualization Path: {visualization_path}")
            self.logger.info(f"🎯 Overlay Results: {len(overlay_results)}")
            self.logger.info(f"🤖 Bot Integration: {bot_integration_path}")
            self.logger.info("=" * 80)
            self.logger.info("✅ Your trading system is now equipped with regime intelligence!")
            self.logger.info("✅ All results are organized by strategy and data file")
            self.logger.info("✅ Bot integration files are ready for deployment")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive analysis failed: {e}")
            raise

async def main():
    """Main function"""
    print("🚀 Comprehensive Regime Analysis and Backtesting System")
    print("=" * 80)
    print("This system will:")
    print("1. 🔍 Analyze historical data to identify market regimes")
    print("2. 🔧 Run backtesting with regime-aware strategies")
    print("3. 📊 Generate comprehensive visualizations with regime overlays")
    print("4. 🤖 Provide regime intelligence for bot decision-making")
    print("5. 📁 Organize all results by strategy and data file")
    print("=" * 80)
    
    # Create and run comprehensive analysis
    analysis = ComprehensiveRegimeAnalysis()
    await analysis.run_comprehensive_analysis()

if __name__ == "__main__":
    asyncio.run(main())
