#!/usr/bin/env python3
"""
Test script for the new engine structure
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all the new modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from core.engine_registry import EngineRegistry, discover_engines
        print("✅ Core imports successful")
        
        # Test config imports
        from configs.engine_config import EngineConfig
        from configs.basic_config import BasicEngineConfig
        from configs.risk_config import RiskManagedEngineConfig
        print("✅ Config imports successful")
        
        # Test main runner
        from engine_runner import EngineRunner
        print("✅ Engine runner import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configs():
    """Test configuration creation and validation"""
    print("\n🧪 Testing configurations...")
    
    try:
        from configs.engine_config import EngineConfig
        from configs.basic_config import BasicEngineConfig
        from configs.risk_config import RiskManagedEngineConfig
        
        # Test base config
        base_config = EngineConfig()
        print(f"✅ Base config created: {base_config.data_path}")
        
        # Test basic config
        basic_config = BasicEngineConfig()
        print(f"✅ Basic config created: {basic_config.save_json}")
        
        # Test risk config
        risk_config = RiskManagedEngineConfig()
        print(f"✅ Risk config created: {risk_config.stop_loss_pct}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_registry():
    """Test the engine registry"""
    print("\n🧪 Testing engine registry...")
    
    try:
        from core.engine_registry import EngineRegistry
        
        # Create registry
        registry = EngineRegistry()
        print("✅ Registry created")
        
        # List engines (should be empty for now since we haven't moved the engine files)
        engines = registry.list_engines()
        print(f"✅ Registry lists engines: {engines}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_runner():
    """Test the engine runner"""
    print("\n🧪 Testing engine runner...")
    
    try:
        from engine_runner import EngineRunner
        
        # Create runner
        runner = EngineRunner()
        print("✅ Engine runner created")
        
        # List available engines
        runner.list_available_engines()
        
        return True
        
    except Exception as e:
        print(f"❌ Runner test failed: {e}")
        return False

def test_directory_structure():
    """Test that the new directory structure exists"""
    print("\n🧪 Testing directory structure...")
    
    base_dir = Path(__file__).parent
    
    required_dirs = [
        'core',
        'engines', 
        'configs',
        'utils',
        'integrations',
        'templates',
        'examples',
        'tests',
        'docs'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            all_exist = False
    
    # Check key files
    key_files = [
        'engine_runner.py',
        'requirements.txt',
        'core/base_engine.py',
        'core/engine_registry.py',
        'configs/engine_config.py',
        'docs/README.md'
    ]
    
    for file_path in key_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🚀 Testing New Engine Structure")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_configs,
        test_registry,
        test_runner
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
        print("🎉 All tests passed! New structure is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Move existing engine files to the engines/ directory")
        print("2. Update imports in existing engines to use new structure")
        print("3. Test with actual backtesting")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
