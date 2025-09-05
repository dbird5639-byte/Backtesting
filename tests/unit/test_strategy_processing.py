#!/usr/bin/env python3
"""
Test strategy processing to identify issues
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_strategy_processing():
    """Test strategy processing step by step"""
    try:
        from engines.basic_engine import BasicEngine
        
        # Create engine
        engine = BasicEngine()
        logger.info("✓ Engine created")
        
        # Discover strategies
        strategies = engine.discover_strategies()
        logger.info(f"✓ Discovered {len(strategies)} strategies")
        
        # Discover data files
        data_files = engine.discover_data_files()
        logger.info(f"✓ Discovered {len(data_files)} data files")
        
        if not data_files:
            logger.error("✗ No data files found - this is the problem!")
            return False
        
        if not strategies:
            logger.error("✗ No strategies found - this is the problem!")
            return False
        
        # Test with first strategy and first data file
        test_strategy = strategies[0]
        test_data = data_files[0]
        
        logger.info(f"Testing strategy: {test_strategy}")
        logger.info(f"Testing data: {test_data}")
        
        # Try to load strategy class
        try:
            strategy_class = engine.load_strategy_class(test_strategy)
            logger.info(f"✓ Strategy class loaded: {strategy_class}")
        except Exception as e:
            logger.error(f"✗ Failed to load strategy class: {e}")
            return False
        
        # Try to run backtest
        try:
            logger.info("Attempting to run backtest...")
            result = engine.run_backtest(test_strategy, test_data)
            logger.info(f"✓ Backtest completed: {result}")
            return True
        except Exception as e:
            logger.error(f"✗ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("Testing strategy processing...")
    
    success = test_strategy_processing()
    
    if success:
        logger.info("✓ Strategy processing test passed!")
    else:
        logger.error("✗ Strategy processing test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
