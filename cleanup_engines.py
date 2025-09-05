#!/usr/bin/env python3
"""
Engine Cleanup Script
Moves old engines to storage and renames improved engines to clean names
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main cleanup function"""
    logger.info("🧹 Starting Engine Cleanup...")
    
    # Define paths
    engines_dir = Path("Engines")
    storage_dir = Path("storage")
    root_dir = Path(".")
    
    # Create storage directory if it doesn't exist
    storage_dir.mkdir(exist_ok=True)
    
    # Create engines_storage subdirectory
    engines_storage = storage_dir / "engines_storage"
    engines_storage.mkdir(exist_ok=True)
    
    # Files to move to storage (old engines)
    old_engines = [
        "advanced_backtesting_engine.py",
        "core_engine.py",
        "enhanced_regime_detection_engine.py", 
        "enhanced_risk_engine.py",
        "enhanced_visualization_engine.py",
        "fibonacci_gann_engine.py",
        "fibonacci_gann_example.py",
        "ml_engine.py",
        "performance_engine.py",
        "portfolio_engine.py",
        "regime_detection_engine.py",
        "regime_overlay_engine.py",
        "regime_visualization_engine.py",
        "risk_engine.py",
        "validation_engine.py"
    ]
    
    # Move old engines to storage
    logger.info("📦 Moving old engines to storage...")
    for engine_file in old_engines:
        source_path = engines_dir / engine_file
        if source_path.exists():
            dest_path = engines_storage / engine_file
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"   ✅ Moved {engine_file} to storage")
        else:
            logger.warning(f"   ⚠️  {engine_file} not found")
    
    # Move engine_factory.py to storage (keeping for reference)
    factory_source = engines_dir / "engine_factory.py"
    if factory_source.exists():
        factory_dest = engines_storage / "engine_factory.py"
        shutil.move(str(factory_source), str(factory_dest))
        logger.info("   ✅ Moved engine_factory.py to storage")
    
    # Move README.md to storage (keeping for reference)
    readme_source = engines_dir / "README.md"
    if readme_source.exists():
        readme_dest = engines_storage / "README.md"
        shutil.move(str(readme_source), str(readme_dest))
        logger.info("   ✅ Moved README.md to storage")
    
    # Move improved engines from root to Engines directory and rename
    improved_engines = {
        "improved_core_engine.py": "core_engine.py",
        "improved_risk_engine.py": "risk_engine.py", 
        "improved_statistical_engine.py": "statistical_engine.py",
        "improved_pipeline_engine.py": "pipeline_engine.py"
    }
    
    logger.info("🔄 Moving and renaming improved engines...")
    for source_name, dest_name in improved_engines.items():
        source_path = root_dir / source_name
        dest_path = engines_dir / dest_name
        
        if source_path.exists():
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"   ✅ Moved and renamed {source_name} → {dest_name}")
        else:
            logger.warning(f"   ⚠️  {source_name} not found")
    
    # Create new README.md for the cleaned engines
    new_readme_content = """# Backtesting Engines

This directory contains the latest, improved backtesting engines based on successful patterns from the old_engines directory.

## Available Engines

### 1. Core Engine (`core_engine.py`)
- **Purpose**: Fundamental backtesting with comprehensive statistics and quality assessment
- **Features**: 
  - Parallel processing with ThreadPoolExecutor
  - Quality assessment scoring system
  - Significance testing with permutation tests
  - Robust error handling and graceful shutdown
  - Auto-resume functionality
  - Data truncation for oversized datasets

### 2. Risk Engine (`risk_engine.py`)
- **Purpose**: Advanced backtesting with comprehensive risk management features
- **Features**:
  - Advanced risk management with position sizing
  - Walkforward optimization for parameter tuning
  - Kelly Criterion position sizing
  - Multiple risk metrics (drawdown, consecutive losses)
  - Risk-adjusted performance analysis
  - Parameter optimization with grid search

### 3. Statistical Engine (`statistical_engine.py`)
- **Purpose**: Advanced statistical analysis and validation of trading strategies
- **Features**:
  - Permutation tests for significance validation
  - Bootstrap analysis for confidence intervals
  - Monte Carlo simulations including GBM
  - Market regime detection with K-means clustering
  - Statistical validation metrics
  - Comprehensive analysis reporting

### 4. Pipeline Engine (`pipeline_engine.py`)
- **Purpose**: Orchestrates multiple engines in sequence with comprehensive result analysis
- **Features**:
  - Sequential engine execution with proper ordering
  - Cross-engine result combination
  - Targeted analysis capabilities
  - Comprehensive reporting with summaries
  - Result aggregation across engines
  - Performance monitoring

## Usage

### Individual Engine Usage
```python
from Engines.core_engine import ImprovedCoreEngine, ImprovedEngineConfig

config = ImprovedEngineConfig()
engine = ImprovedCoreEngine(config)
engine.run()
```

### Pipeline Usage
```python
from Engines.pipeline_engine import ImprovedPipelineEngine, PipelineEngineConfig

config = PipelineEngineConfig()
engine = ImprovedPipelineEngine(config)
engine.run()
```

## Configuration

Each engine has its own configuration class with specialized parameters. See individual engine files for detailed configuration options.

## Results

Results are saved to the `Results/` directory with timestamps and organized by engine type.

## Old Engines

Previous versions of engines have been moved to `storage/engines_storage/` for reference.
"""
    
    new_readme_path = engines_dir / "README.md"
    with open(new_readme_path, 'w') as f:
        f.write(new_readme_content)
    logger.info("   ✅ Created new README.md")
    
    # Create __init__.py for the engines package
    init_content = """# Backtesting Engines Package

This package contains the latest, improved backtesting engines.

## Available Engines

- Core Engine: Fundamental backtesting with quality assessment
- Risk Engine: Advanced risk management and walkforward optimization  
- Statistical Engine: Statistical validation and regime analysis
- Pipeline Engine: Orchestrates multiple engines in sequence

## Usage

```python
from Engines.core_engine import ImprovedCoreEngine
from Engines.risk_engine import ImprovedRiskEngine
from Engines.statistical_engine import ImprovedStatisticalEngine
from Engines.pipeline_engine import ImprovedPipelineEngine
```
"""
    
    init_path = engines_dir / "__init__.py"
    with open(init_path, 'w') as f:
        f.write(init_content)
    logger.info("   ✅ Updated __init__.py")
    
    # Clean up __pycache__ directories
    logger.info("🧹 Cleaning up __pycache__ directories...")
    for pycache_dir in engines_dir.rglob("__pycache__"):
        shutil.rmtree(pycache_dir)
        logger.info(f"   ✅ Removed {pycache_dir}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎉 ENGINE CLEANUP COMPLETE!")
    logger.info("="*60)
    logger.info("✅ Moved old engines to storage/engines_storage/")
    logger.info("✅ Renamed improved engines to clean names")
    logger.info("✅ Created new README.md and __init__.py")
    logger.info("✅ Cleaned up __pycache__ directories")
    logger.info("\n📁 Current Engines directory contains:")
    
    # List current engines directory contents
    for item in sorted(engines_dir.iterdir()):
        if item.is_file():
            logger.info(f"   📄 {item.name}")
        elif item.is_dir():
            logger.info(f"   📁 {item.name}/")
    
    logger.info("\n📦 Old engines moved to storage/engines_storage/")
    for item in sorted(engines_storage.iterdir()):
        if item.is_file():
            logger.info(f"   📄 {item.name}")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()
