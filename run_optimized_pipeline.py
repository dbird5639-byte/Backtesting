#!/usr/bin/env python3
"""
Optimized Pipeline Runner - Runs all engines in correct order with regime engines last
High performance, reliability, and comprehensive results
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add Engines directory to path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

# Import engines
from Engines.core_engine import CoreEngine, EngineConfig
from Engines.risk_engine import RiskEngine, RiskEngineConfig
from Engines.statistical_engine import StatisticalEngine, StatisticalEngineConfig
from Engines.validation_engine import ValidationEngine, ValidationEngineConfig
from Engines.portfolio_engine import PortfolioEngine, PortfolioEngineConfig
from Engines.ml_engine import MLEngine, MLEngineConfig
from Engines.performance_engine import PerformanceEngine, PerformanceEngineConfig
from Engines.regime_detection_engine import RegimeDetectionEngine, RegimeDetectionConfig
from Engines.regime_overlay_engine import RegimeOverlayEngine, RegimeOverlayConfig
from Engines.regime_visualization_engine import RegimeVisualizationEngine, RegimeVisualizationConfig
from Engines.enhanced_visualization_engine import EnhancedVisualizationEngine, EnhancedVisualizationConfig
from Engines.fibonacci_gann_engine import FibonacciGannEngine, FibonacciGannConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pipeline_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class OptimizedPipelineRunner:
    """Optimized pipeline runner with proper engine ordering"""
    
    def __init__(self):
        self.logger = logger
        self.start_time = None
        self.results = {}
        self.engine_order = self._get_engine_order()
        
    def _get_engine_order(self) -> List[Dict[str, Any]]:
        """Get engines in correct execution order with regime engines last"""
        return [
            # Core engines first
            {
                'name': 'CoreEngine',
                'class': CoreEngine,
                'config_class': EngineConfig,
                'priority': 1,
                'description': 'Fundamental backtesting with quality assessment'
            },
            {
                'name': 'RiskEngine', 
                'class': RiskEngine,
                'config_class': RiskEngineConfig,
                'priority': 2,
                'description': 'Advanced risk management and walkforward optimization'
            },
            {
                'name': 'StatisticalEngine',
                'class': StatisticalEngine,
                'config_class': StatisticalEngineConfig,
                'priority': 3,
                'description': 'Statistical validation and regime analysis'
            },
            {
                'name': 'ValidationEngine',
                'class': ValidationEngine,
                'config_class': ValidationEngineConfig,
                'priority': 4,
                'description': 'Comprehensive validation with multiple testing methods'
            },
            
            # Analysis engines
            {
                'name': 'PortfolioEngine',
                'class': PortfolioEngine,
                'config_class': PortfolioEngineConfig,
                'priority': 5,
                'description': 'Multi-objective portfolio optimization'
            },
            {
                'name': 'MLEngine',
                'class': MLEngine,
                'config_class': MLEngineConfig,
                'priority': 6,
                'description': 'Machine learning-powered backtesting'
            },
            {
                'name': 'PerformanceEngine',
                'class': PerformanceEngine,
                'config_class': PerformanceEngineConfig,
                'priority': 7,
                'description': 'Advanced performance analytics'
            },
            
            # Visualization engines
            {
                'name': 'EnhancedVisualizationEngine',
                'class': EnhancedVisualizationEngine,
                'config_class': EnhancedVisualizationConfig,
                'priority': 8,
                'description': 'Comprehensive visualization capabilities'
            },
            {
                'name': 'FibonacciGannEngine',
                'class': FibonacciGannEngine,
                'config_class': FibonacciGannConfig,
                'priority': 9,
                'description': 'Advanced Fibonacci and Gann analysis'
            },
            
            # Regime engines LAST (as requested)
            {
                'name': 'RegimeDetectionEngine',
                'class': RegimeDetectionEngine,
                'config_class': RegimeDetectionConfig,
                'priority': 10,
                'description': 'Market regime identification and analysis'
            },
            {
                'name': 'RegimeOverlayEngine',
                'class': RegimeOverlayEngine,
                'config_class': RegimeOverlayConfig,
                'priority': 11,
                'description': 'Regime overlay for existing results'
            },
            {
                'name': 'RegimeVisualizationEngine',
                'class': RegimeVisualizationEngine,
                'config_class': RegimeVisualizationConfig,
                'priority': 12,
                'description': 'Regime visualization and analysis'
            }
        ]
    
    def run_engine(self, engine_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single engine with comprehensive error handling"""
        engine_name = engine_info['name']
        engine_class = engine_info['class']
        config_class = engine_info['config_class']
        description = engine_info['description']
        
        try:
            self.logger.info(f"🚀 Starting {engine_name} - {description}")
            start_time = time.time()
            
            # Create engine instance
            config = config_class()
            engine = engine_class(config)
            
            # Run engine
            engine.run()
            
            execution_time = time.time() - start_time
            
            result = {
                'engine_name': engine_name,
                'description': description,
                'execution_time': execution_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ {engine_name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error running {engine_name}: {e}")
            return {
                'engine_name': engine_name,
                'description': description,
                'execution_time': 0,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_pipeline(self):
        """Run the complete pipeline in correct order"""
        self.start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING OPTIMIZED PIPELINE RUNNER")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Total engines to run: {len(self.engine_order)}")
        self.logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        
        # Run engines in order
        for i, engine_info in enumerate(self.engine_order, 1):
            self.logger.info(f"\n📈 Engine {i}/{len(self.engine_order)}: {engine_info['name']}")
            self.logger.info(f"   Description: {engine_info['description']}")
            self.logger.info(f"   Priority: {engine_info['priority']}")
            
            result = self.run_engine(engine_info)
            self.results[engine_info['name']] = result
            
            # Show progress
            completed = len([r for r in self.results.values() if r['status'] == 'completed'])
            failed = len([r for r in self.results.values() if r['status'] == 'failed'])
            total_time = time.time() - self.start_time
            
            self.logger.info(f"   📊 Progress: {completed} completed, {failed} failed")
            self.logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
            
            # Add separator for regime engines
            if engine_info['priority'] == 10:
                self.logger.info("\n" + "=" * 60)
                self.logger.info("🎯 STARTING REGIME ENGINES (LAST IN LINE)")
                self.logger.info("=" * 60)
        
        # Final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final pipeline summary"""
        total_time = time.time() - self.start_time
        completed = len([r for r in self.results.values() if r['status'] == 'completed'])
        failed = len([r for r in self.results.values() if r['status'] == 'failed'])
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎉 PIPELINE EXECUTION COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"⏰ Total execution time: {total_time:.2f}s ({total_time/60:.1f}m)")
        self.logger.info(f"✅ Engines completed: {completed}/{len(self.engine_order)}")
        self.logger.info(f"❌ Engines failed: {failed}/{len(self.engine_order)}")
        self.logger.info(f"📊 Success rate: {completed/len(self.engine_order)*100:.1f}%")
        
        # Engine details
        self.logger.info("\n📋 Engine Results:")
        for engine_name, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'completed' else "❌"
            self.logger.info(f"   {status_emoji} {engine_name}: {result['execution_time']:.2f}s - {result['status']}")
        
        # Regime engines summary
        regime_engines = [name for name in self.results.keys() if 'Regime' in name]
        if regime_engines:
            self.logger.info(f"\n🎯 Regime Engines (Last in Line): {len(regime_engines)}")
            for engine_name in regime_engines:
                result = self.results[engine_name]
                status_emoji = "✅" if result['status'] == 'completed' else "❌"
                self.logger.info(f"   {status_emoji} {engine_name}: {result['execution_time']:.2f}s")
        
        self.logger.info("=" * 80)
        
        # Save pipeline results
        self._save_pipeline_results()
    
    def _save_pipeline_results(self):
        """Save pipeline execution results"""
        try:
            results_dir = Path("Results") / "Pipeline" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save pipeline results
            pipeline_summary = {
                'pipeline_start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'pipeline_end_time': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.start_time,
                'total_engines': len(self.engine_order),
                'completed_engines': len([r for r in self.results.values() if r['status'] == 'completed']),
                'failed_engines': len([r for r in self.results.values() if r['status'] == 'failed']),
                'success_rate': len([r for r in self.results.values() if r['status'] == 'completed']) / len(self.engine_order),
                'engine_results': self.results
            }
            
            # Save JSON
            json_path = results_dir / "pipeline_results.json"
            with open(json_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
            
            self.logger.info(f"📁 Pipeline results saved to: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline results: {e}")

def main():
    """Main entry point"""
    try:
        runner = OptimizedPipelineRunner()
        runner.run_pipeline()
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
