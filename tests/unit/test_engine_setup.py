#!/usr/bin/env python3
"""
Test script to verify engine setup and strategy discovery
"""

import os
import sys
from pathlib import Path

# Add the Engines directory to Python path
engines_dir = Path(__file__).parent / "Engines"
sys.path.insert(0, str(engines_dir))

def test_directory_structure():
    """Test that all required directories exist"""
    print("🔍 Testing directory structure...")
    
    base_path = Path(r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting")
    
    required_dirs = [
        base_path / "Data" / "Hyperliquid",
        base_path / "Strategies" / "storage", 
        base_path / "Results"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist

def test_strategy_discovery():
    """Test that strategies can be discovered"""
    print("\n🔍 Testing strategy discovery...")
    
    try:
        from base_engine import BaseEngine, EngineConfig
        
        # Create a test config
        config = EngineConfig()
        engine = BaseEngine(config)
        
        # Discover strategies
        strategies = engine.discover_strategies()
        
        if strategies:
            print(f"✅ Found {len(strategies)} strategies:")
            for i, strategy in enumerate(strategies[:5]):  # Show first 5
                print(f"   {i+1}. {os.path.basename(strategy)}")
            if len(strategies) > 5:
                print(f"   ... and {len(strategies) - 5} more")
        else:
            print("❌ No strategies found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error discovering strategies: {e}")
        return False

def test_data_discovery():
    """Test that data files can be discovered"""
    print("\n🔍 Testing data discovery...")
    
    try:
        from base_engine import BaseEngine, EngineConfig
        
        # Create a test config
        config = EngineConfig()
        engine = BaseEngine(config)
        
        # Discover data files
        data_files = engine.discover_data_files()
        
        if data_files:
            print(f"✅ Found {len(data_files)} data files:")
            for i, data_file in enumerate(data_files[:5]):  # Show first 5
                print(f"   {i+1}. {os.path.basename(data_file)}")
            if len(data_files) > 5:
                print(f"   ... and {len(data_files) - 5} more")
        else:
            print("❌ No data files found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error discovering data files: {e}")
        return False

def test_strategy_loading():
    """Test that a strategy can be loaded"""
    print("\n🔍 Testing strategy loading...")
    
    try:
        from base_engine import BaseEngine, EngineConfig
        
        # Create a test config
        config = EngineConfig()
        engine = BaseEngine(config)
        
        # Discover strategies
        strategies = engine.discover_strategies()
        
        if not strategies:
            print("❌ No strategies to test loading")
            return False
        
        # Try to load the first strategy
        test_strategy = strategies[0]
        print(f"Testing strategy: {os.path.basename(test_strategy)}")
        
        strategy_class = engine.load_strategy(test_strategy)
        
        if strategy_class:
            print(f"✅ Successfully loaded strategy class: {strategy_class.__name__}")
            return True
        else:
            print("❌ Failed to load strategy class")
            return False
            
    except Exception as e:
        print(f"❌ Error loading strategy: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Engine Setup Test Suite")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_strategy_discovery,
        test_data_discovery,
        test_strategy_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your engine setup is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
