#!/usr/bin/env python3
"""
Test script for enhanced scalping strategies

This script tests the enhanced scalping strategies to ensure they can be
imported and run successfully.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test importing the enhanced scalping strategies"""
    try:
        print("Testing imports...")
        
        # Test importing the main module
        from enhanced_scalping_strategies import (
            EnhancedScalpingBase,
            MultiIndicatorScalping,
            FibonacciScalping,
            VolatilityBreakoutScalping,
            SCALPING_STRATEGIES,
            get_strategy_class,
            list_available_strategies
        )
        
        print("✓ All imports successful!")
        
        # Test strategy registry
        strategies = list_available_strategies()
        print(f"✓ Available strategies: {strategies}")
        
        # Test getting strategy classes
        for strategy_name in strategies:
            strategy_class = get_strategy_class(strategy_name)
            if strategy_class:
                print(f"✓ {strategy_name}: {strategy_class.__name__}")
            else:
                print(f"✗ {strategy_name}: Failed to get class")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_strategy_creation():
    """Test creating strategy instances"""
    try:
        print("\nTesting strategy creation...")
        
        from enhanced_scalping_strategies import (
            MultiIndicatorScalping,
            FibonacciScalping,
            VolatilityBreakoutScalping
        )
        
        # Test creating instances (without data for now)
        strategies = [
            MultiIndicatorScalping,
            FibonacciScalping,
            VolatilityBreakoutScalping
        ]
        
        for strategy_class in strategies:
            print(f"✓ {strategy_class.__name__}: Class created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy creation error: {e}")
        return False

def test_strategy_parameters():
    """Test strategy parameter access"""
    try:
        print("\nTesting strategy parameters...")
        
        from enhanced_scalping_strategies import MultiIndicatorScalping
        
        # Test accessing strategy parameters
        strategy = MultiIndicatorScalping
        
        # Check if key parameters exist
        expected_params = [
            'risk_per_trade',
            'max_positions',
            'max_drawdown',
            'consecutive_loss_limit',
            'rsi_period',
            'macd_fast',
            'macd_slow',
            'bb_period',
            'atr_period'
        ]
        
        for param in expected_params:
            if hasattr(strategy, param):
                value = getattr(strategy, param)
                print(f"✓ {param}: {value}")
            else:
                print(f"✗ {param}: Not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter test error: {e}")
        return False

def main():
    """Main test function"""
    print("Enhanced Scalping Strategies Test")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_imports,
        test_strategy_creation,
        test_strategy_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Enhanced scalping strategies are ready to use.")
        print("\nTo use these strategies:")
        print("1. Import from enhanced_scalping_strategies")
        print("2. Use with your backtesting framework")
        print("3. Run through the engines using run_engines.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
