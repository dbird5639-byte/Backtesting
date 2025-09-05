#!/usr/bin/env python3
"""
Enhanced Risk Analysis Script with Walk-Forward Testing

This script runs comprehensive risk analysis with walk-forward testing,
comparing safe vs risky parameter values across in-sample and out-of-sample periods.

Features:
- Walk-forward analysis with configurable in-sample and out-of-sample periods
- Five safe parameter values (conservative risk management)
- Five risky parameter values (aggressive risk management)
- Comprehensive parameter comparison and optimization
- Risk attribution and decomposition analysis
- Integration with existing regime analysis and visualization engines
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
from Engines.enhanced_risk_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig, RiskLevel, WalkForwardPhase
from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, VisualizationConfig
from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig

class EnhancedRiskAnalysisOrchestrator:
    """Orchestrates comprehensive enhanced risk analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        
        # Initialize configurations
        self.risk_config = EnhancedRiskEngineConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            walk_forward_enabled=True,
            in_sample_periods=252,  # 1 year in-sample
            out_of_sample_periods=63,  # 3 months out-of-sample
            min_periods_for_analysis=100,
            save_walk_forward_results=True,
            save_parameter_comparison=True,
            save_risk_attribution=True,
            create_visualizations=True
        )
        
        self.visualization_config = VisualizationConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            save_csv=True,
            save_png=True,
            save_html=True,
            enable_regime_overlay=True,
            organize_by_strategy=True,
            organize_by_data_file=True,
            create_summary_dashboard=True
        )
        
        self.regime_overlay_config = RegimeOverlayConfig(
            data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data",
            strategies_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Strategies",
            results_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            regime_data_path=r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results",
            enable_regime_filtering=True,
            enable_strategy_recommendations=True,
            enable_regime_alerts=True
        )
        
        # Initialize engines
        self.risk_engine = EnhancedRiskEngine(self.risk_config)
        self.visualization_engine = EnhancedVisualizationEngine(self.visualization_config)
        self.regime_overlay_engine = RegimeOverlayEngine(self.regime_overlay_config)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'enhanced_risk_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def discover_data_files(self) -> List[str]:
        """Discover all data files"""
        data_path = Path(self.risk_config.data_path)
        if not data_path.exists():
            self.logger.warning(f"Data path does not exist: {data_path}")
            return []
        
        csv_files = list(data_path.rglob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files")
        return [str(f) for f in csv_files]
    
    async def run_enhanced_risk_analysis(self, data_files: List[str]) -> Dict[str, Any]:
        """Run comprehensive enhanced risk analysis"""
        self.logger.info("Starting Enhanced Risk Analysis with Walk-Forward Testing")
        
        try:
            # Run walk-forward risk analysis
            risk_results = await self.risk_engine.run_walk_forward_analysis(data_files)
            
            self.logger.info(f"Enhanced risk analysis complete: {len(risk_results.get('walk_forward_results', []))} results")
            return risk_results
            
        except Exception as e:
            self.logger.error(f"Enhanced risk analysis failed: {e}")
            return {}
    
    def create_risk_visualizations(self, risk_results: Dict[str, Any]) -> str:
        """Create comprehensive risk visualizations"""
        self.logger.info("Creating enhanced risk visualizations...")
        
        try:
            # Convert risk results to visualization format
            visualization_data = self._convert_risk_results_to_visualization_format(risk_results)
            
            # Create visualizations
            visualization_path = asyncio.run(
                self.visualization_engine.run_visualization_analysis(visualization_data)
            )
            
            self.logger.info(f"Risk visualizations created: {visualization_path}")
            return visualization_path
            
        except Exception as e:
            self.logger.error(f"Risk visualization creation failed: {e}")
            return ""
    
    def _convert_risk_results_to_visualization_format(self, risk_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert risk results to visualization format"""
        visualization_data = []
        
        walk_forward_results = risk_results.get('walk_forward_results', [])
        
        for result in walk_forward_results:
            # Create visualization entry
            viz_entry = {
                'strategy_name': f"{result.risk_level.value}_{result.phase.value}",
                'data_file': result.metadata.get('data_file', 'Unknown'),
                'total_return': result.performance_metrics.get('total_return', 0),
                'sharpe_ratio': result.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': result.performance_metrics.get('max_drawdown', 0),
                'win_rate': result.performance_metrics.get('win_rate', 0),
                'volatility': result.performance_metrics.get('volatility', 0),
                'risk_level': result.risk_level.value,
                'phase': result.phase.value,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'parameters': result.parameters,
                'metadata': {
                    'trades_count': result.trades_count,
                    'analysis_timestamp': result.analysis_timestamp
                }
            }
            visualization_data.append(viz_entry)
        
        return visualization_data
    
    def create_parameter_comparison_report(self, risk_results: Dict[str, Any]) -> str:
        """Create comprehensive parameter comparison report"""
        self.logger.info("📋 Creating parameter comparison report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"./Results/EnhancedRiskEngine/parameter_comparison_report_{timestamp}"
            os.makedirs(report_path, exist_ok=True)
            
            parameter_comparisons = risk_results.get('parameter_comparisons', [])
            summary_statistics = risk_results.get('summary_statistics', {})
            
            # Create detailed report
            report_file = f"{report_path}/detailed_comparison_report.txt"
            with open(report_file, 'w') as f:
                f.write("Enhanced Risk Engine - Parameter Comparison Report\n")
                f.write("=" * 60 + "\n\n")
                
                # Summary statistics
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Walk-Forward Windows: {summary_statistics.get('total_windows', 0)}\n")
                f.write(f"Safe In-Sample Results: {summary_statistics.get('safe_in_sample_count', 0)}\n")
                f.write(f"Safe Out-of-Sample Results: {summary_statistics.get('safe_out_sample_count', 0)}\n")
                f.write(f"Risky In-Sample Results: {summary_statistics.get('risky_in_sample_count', 0)}\n")
                f.write(f"Risky Out-of-Sample Results: {summary_statistics.get('risky_out_sample_count', 0)}\n\n")
                
                # Parameter comparisons
                f.write("PARAMETER COMPARISONS\n")
                f.write("-" * 25 + "\n")
                for comparison in parameter_comparisons:
                    f.write(f"\n{comparison.parameter_name.upper()}:\n")
                    f.write(f"  Safe Value: {comparison.safe_value}\n")
                    f.write(f"  Risky Value: {comparison.risky_value}\n")
                    f.write(f"  Recommendation: {comparison.recommendation}\n")
                    
                    # Performance differences
                    f.write("  Performance Differences (Risky - Safe):\n")
                    for metric, diff in comparison.performance_difference.items():
                        f.write(f"    {metric}: {diff:+.4f}\n")
                
                # Performance summary
                f.write("\n\nPERFORMANCE SUMMARY\n")
                f.write("-" * 20 + "\n")
                perf_summary = summary_statistics.get('performance_summary', {})
                for group_name, group_perf in perf_summary.items():
                    f.write(f"\n{group_name.replace('_', ' ').title()}:\n")
                    for metric, stats in group_perf.items():
                        f.write(f"  {metric}: {stats['mean']:.4f} (std: {stats['std']:.4f})\n")
            
            # Create JSON summary
            json_file = f"{report_path}/comparison_summary.json"
            comparison_summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_parameters_compared': len(parameter_comparisons),
                'parameter_recommendations': [
                    {
                        'parameter': comp.parameter_name,
                        'safe_value': comp.safe_value,
                        'risky_value': comp.risky_value,
                        'recommendation': comp.recommendation,
                        'key_differences': {
                            metric: diff for metric, diff in comp.performance_difference.items()
                            if abs(diff) > 0.01  # Only significant differences
                        }
                    }
                    for comp in parameter_comparisons
                ],
                'summary_statistics': summary_statistics
            }
            
            with open(json_file, 'w') as f:
                json.dump(comparison_summary, f, indent=2)
            
            self.logger.info(f"✅ Parameter comparison report created: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Parameter comparison report creation failed: {e}")
            return ""
    
    def create_risk_attribution_report(self, risk_results: Dict[str, Any]) -> str:
        """Create risk attribution analysis report"""
        self.logger.info("🎯 Creating risk attribution report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"./Results/EnhancedRiskEngine/risk_attribution_report_{timestamp}"
            os.makedirs(report_path, exist_ok=True)
            
            risk_attribution = risk_results.get('risk_attribution', {})
            
            # Create detailed attribution report
            report_file = f"{report_path}/risk_attribution_analysis.txt"
            with open(report_file, 'w') as f:
                f.write("Enhanced Risk Engine - Risk Attribution Analysis\n")
                f.write("=" * 50 + "\n\n")
                
                # Parameter impact analysis
                f.write("PARAMETER IMPACT ANALYSIS\n")
                f.write("-" * 30 + "\n")
                param_impact = risk_attribution.get('parameter_impact', {})
                for param, impact_data in param_impact.items():
                    f.write(f"\n{param.upper()}:\n")
                    for value, performance in impact_data.items():
                        f.write(f"  Value {value}:\n")
                        for metric, perf_value in performance.items():
                            f.write(f"    {metric}: {perf_value:.4f}\n")
                
                # Phase impact analysis
                f.write("\n\nPHASE IMPACT ANALYSIS\n")
                f.write("-" * 25 + "\n")
                phase_impact = risk_attribution.get('phase_impact', {})
                for phase, performance in phase_impact.items():
                    if phase != 'degradation':
                        f.write(f"\n{phase.replace('_', ' ').title()}:\n")
                        for metric, perf_value in performance.items():
                            f.write(f"  {metric}: {perf_value:.4f}\n")
                
                # Degradation analysis
                if 'degradation' in phase_impact:
                    f.write("\nPERFORMANCE DEGRADATION (Out-of-Sample vs In-Sample):\n")
                    for metric, degradation in phase_impact['degradation'].items():
                        f.write(f"  {metric}: {degradation:+.2%}\n")
                
                # Risk level impact
                f.write("\n\nRISK LEVEL IMPACT ANALYSIS\n")
                f.write("-" * 30 + "\n")
                risk_level_impact = risk_attribution.get('risk_level_impact', {})
                for risk_level, performance in risk_level_impact.items():
                    if risk_level != 'tradeoff':
                        f.write(f"\n{risk_level.upper()}:\n")
                        for metric, perf_value in performance.items():
                            f.write(f"  {metric}: {perf_value:.4f}\n")
                
                # Risk-return trade-off
                if 'tradeoff' in risk_level_impact:
                    f.write("\nRISK-RETURN TRADE-OFF (Risky - Safe):\n")
                    for metric, tradeoff in risk_level_impact['tradeoff'].items():
                        f.write(f"  {metric}: {tradeoff:+.4f}\n")
            
            # Save detailed attribution data
            json_file = f"{report_path}/risk_attribution_data.json"
            with open(json_file, 'w') as f:
                json.dump(risk_attribution, f, indent=2)
            
            self.logger.info(f"✅ Risk attribution report created: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Risk attribution report creation failed: {e}")
            return ""
    
    def create_executive_summary(self, risk_results: Dict[str, Any]) -> str:
        """Create executive summary of enhanced risk analysis"""
        self.logger.info("📊 Creating executive summary...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = f"./Results/EnhancedRiskEngine/executive_summary_{timestamp}"
            os.makedirs(summary_path, exist_ok=True)
            
            parameter_comparisons = risk_results.get('parameter_comparisons', [])
            summary_statistics = risk_results.get('summary_statistics', {})
            risk_attribution = risk_results.get('risk_attribution', {})
            
            # Create executive summary
            summary_file = f"{summary_path}/executive_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("ENHANCED RISK ENGINE - EXECUTIVE SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("ANALYSIS OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Walk-Forward Windows: {summary_statistics.get('total_windows', 0)}\n")
                f.write(f"Parameters Analyzed: {len(parameter_comparisons)}\n")
                f.write(f"Risk Levels Tested: Safe vs Risky\n")
                f.write(f"Analysis Phases: In-Sample vs Out-of-Sample\n\n")
                
                f.write("KEY FINDINGS\n")
                f.write("-" * 15 + "\n")
                
                # Top recommendations
                f.write("Top Parameter Recommendations:\n")
                for i, comparison in enumerate(parameter_comparisons[:5], 1):
                    f.write(f"{i}. {comparison.parameter_name}: {comparison.recommendation}\n")
                
                f.write("\nPerformance Summary:\n")
                perf_summary = summary_statistics.get('performance_summary', {})
                
                # Safe vs Risky comparison
                safe_perf = perf_summary.get('safe_in_sample', {})
                risky_perf = perf_summary.get('risky_in_sample', {})
                
                if safe_perf and risky_perf:
                    f.write("  In-Sample Performance (Safe vs Risky):\n")
                    for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                        safe_val = safe_perf.get(metric, {}).get('mean', 0)
                        risky_val = risky_perf.get(metric, {}).get('mean', 0)
                        f.write(f"    {metric}: Safe={safe_val:.4f}, Risky={risky_val:.4f}\n")
                
                # Out-of-sample degradation
                phase_impact = risk_attribution.get('phase_impact', {})
                if 'degradation' in phase_impact:
                    f.write("\n  Out-of-Sample Performance Degradation:\n")
                    for metric, degradation in phase_impact['degradation'].items():
                        f.write(f"    {metric}: {degradation:+.2%}\n")
                
                f.write("\nRECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                f.write("1. Review parameter recommendations above for strategy optimization\n")
                f.write("2. Consider out-of-sample performance degradation in strategy selection\n")
                f.write("3. Balance risk-return trade-offs based on your risk tolerance\n")
                f.write("4. Monitor parameter stability across different market regimes\n")
                f.write("5. Implement dynamic parameter adjustment based on market conditions\n")
            
            # Create JSON executive summary
            json_file = f"{summary_path}/executive_summary.json"
            executive_summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_windows': summary_statistics.get('total_windows', 0),
                'parameters_analyzed': len(parameter_comparisons),
                'top_recommendations': [
                    {
                        'parameter': comp.parameter_name,
                        'recommendation': comp.recommendation,
                        'safe_value': comp.safe_value,
                        'risky_value': comp.risky_value
                    }
                    for comp in parameter_comparisons[:5]
                ],
                'performance_summary': perf_summary,
                'key_insights': {
                    'safe_vs_risky_performance': {
                        'safe_avg_return': safe_perf.get('total_return', {}).get('mean', 0) if safe_perf else 0,
                        'risky_avg_return': risky_perf.get('total_return', {}).get('mean', 0) if risky_perf else 0,
                        'safe_avg_sharpe': safe_perf.get('sharpe_ratio', {}).get('mean', 0) if safe_perf else 0,
                        'risky_avg_sharpe': risky_perf.get('sharpe_ratio', {}).get('mean', 0) if risky_perf else 0
                    },
                    'out_of_sample_degradation': phase_impact.get('degradation', {})
                }
            }
            
            with open(json_file, 'w') as f:
                json.dump(executive_summary, f, indent=2)
            
            self.logger.info(f"✅ Executive summary created: {summary_path}")
            return summary_path
            
        except Exception as e:
            self.logger.error(f"❌ Executive summary creation failed: {e}")
            return ""
    
    async def run_comprehensive_enhanced_risk_analysis(self):
        """Run complete comprehensive enhanced risk analysis"""
        self.logger.info("🚀 Starting Comprehensive Enhanced Risk Analysis")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Discover data files
            data_files = self.discover_data_files()
            if not data_files:
                self.logger.error("No data files found. Exiting.")
                return
            
            # Step 2: Run enhanced risk analysis
            risk_results = await self.run_enhanced_risk_analysis(data_files)
            
            if not risk_results:
                self.logger.error("No risk results generated. Exiting.")
                return
            
            # Step 3: Create visualizations
            visualization_path = self.create_risk_visualizations(risk_results)
            
            # Step 4: Create parameter comparison report
            comparison_report_path = self.create_parameter_comparison_report(risk_results)
            
            # Step 5: Create risk attribution report
            attribution_report_path = self.create_risk_attribution_report(risk_results)
            
            # Step 6: Create executive summary
            executive_summary_path = self.create_executive_summary(risk_results)
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 COMPREHENSIVE ENHANCED RISK ANALYSIS COMPLETE!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Total Duration: {duration}")
            self.logger.info(f"📁 Data Files Processed: {len(data_files)}")
            self.logger.info(f"🔍 Walk-Forward Results: {len(risk_results.get('walk_forward_results', []))}")
            self.logger.info(f"📊 Parameter Comparisons: {len(risk_results.get('parameter_comparisons', []))}")
            self.logger.info(f"📈 Visualizations: {visualization_path}")
            self.logger.info(f"📋 Comparison Report: {comparison_report_path}")
            self.logger.info(f"🎯 Attribution Report: {attribution_report_path}")
            self.logger.info(f"📊 Executive Summary: {executive_summary_path}")
            self.logger.info("=" * 80)
            self.logger.info("✅ Your risk engine now has comprehensive walk-forward analysis!")
            self.logger.info("✅ Safe and risky parameters have been thoroughly tested!")
            self.logger.info("✅ In-sample and out-of-sample performance has been analyzed!")
            self.logger.info("✅ Parameter optimization recommendations are ready!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive enhanced risk analysis failed: {e}")
            raise

async def main():
    """Main function"""
    print("Enhanced Risk Analysis with Walk-Forward Testing")
    print("=" * 80)
    print("This system will:")
    print("1. Run walk-forward analysis with in-sample and out-of-sample testing")
    print("2. Test five safe parameter values (conservative risk management)")
    print("3. Test five risky parameter values (aggressive risk management)")
    print("4. Compare parameter performance across all risk levels and phases")
    print("5. Provide comprehensive risk attribution and decomposition")
    print("6. Generate detailed reports and visualizations")
    print("=" * 80)
    
    # Create and run comprehensive analysis
    orchestrator = EnhancedRiskAnalysisOrchestrator()
    await orchestrator.run_comprehensive_enhanced_risk_analysis()

if __name__ == "__main__":
    asyncio.run(main())
