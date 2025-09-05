#!/usr/bin/env python3
"""
Debug script for BasicEngine to identify issues
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_engine():
    """Test the basic engine step by step"""
    try:
        logger.info("Testing BasicEngine import...")
        from engines.basic_engine import BasicEngine
        logger.info("✓ BasicEngine imported successfully")
        
        logger.info("Testing BasicEngine instantiation...")
        engine = BasicEngine()
        logger.info("✓ BasicEngine instantiated successfully")
        
        logger.info("Testing strategy discovery...")
        strategies = engine.discover_strategies()
        logger.info(f"✓ Discovered {len(strategies)} strategies: {strategies}")
        
        logger.info("Testing data file discovery...")
        data_files = engine.discover_data_files()
        logger.info(f"✓ Discovered {len(data_files)} data files: {data_files}")
        
        if strategies and data_files:
            logger.info("Testing strategy loading...")
            first_strategy = strategies[0]
            logger.info(f"Testing strategy: {first_strategy}")
            
            # Try to load the strategy class
            try:
                strategy_class = engine.load_strategy_class(first_strategy)
                logger.info(f"✓ Strategy class loaded: {strategy_class}")
            except Exception as e:
                logger.error(f"✗ Failed to load strategy class: {e}")
                return False
        
        logger.info("✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_loading():
    """Test loading individual strategies"""
    try:
        logger.info("Testing individual strategy loading...")
        
        # Test the new enhanced scalping strategies
        strategies_to_test = [
            'multi_indicator_scalping',
            'fibonacci_scalping', 
            'volatility_breakout_scalping',
            'momentum_scalping',
            'mean_reversion_scalping'
        ]
        
        for strategy_name in strategies_to_test:
            try:
                strategy_path = Path(__file__).parent.parent / 'Strategies' / f'{strategy_name}.py'
                if strategy_path.exists():
                    logger.info(f"Testing {strategy_name}...")
                    
                    # Try to import the strategy
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find the strategy class
                    strategy_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if hasattr(attr, '__bases__') and any('Strategy' in str(base) for base in attr.__bases__):
                            strategy_class = attr
                            break
                    
                    if strategy_class:
                        logger.info(f"✓ {strategy_name}: {strategy_class.__name__}")
                    else:
                        logger.warning(f"⚠ {strategy_name}: No Strategy class found")
                        
                else:
                    logger.warning(f"⚠ {strategy_name}: File not found")
                    
            except Exception as e:
                logger.error(f"✗ {strategy_name}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Strategy loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("Starting BasicEngine debug tests...")
    
    # Test basic engine
    engine_test = test_basic_engine()
    
    # Test strategy loading
    strategy_test = test_strategy_loading()
    
    # Summary
    logger.info("=" * 50)
    if engine_test and strategy_test:
        logger.info("✓ All tests passed! BasicEngine should work correctly.")
    else:
        logger.error("✗ Some tests failed. Check the logs above for issues.")
    
    return engine_test and strategy_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
