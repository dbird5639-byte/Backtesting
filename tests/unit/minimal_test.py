#!/usr/bin/env python3
"""
Minimal test
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_minimal():
    """Test minimal functionality"""
    print("1. Testing basic imports...")
    
    try:
        print("2. Testing simple base engine import...")
        from simple_base_engine import SimpleBaseEngine
        print("✓ Simple base engine imported")
        
        print("3. Testing simple basic engine import...")
        from simple_basic_engine import SimpleBasicEngine
        print("✓ Simple basic engine imported")
        
        print("4. Testing engine creation...")
        engine = SimpleBasicEngine()
        print("✓ Engine created")
        
        print("5. Testing strategy discovery...")
        strategies = engine.discover_strategies()
        print(f"✓ Found {len(strategies)} strategies")
        
        print("6. Testing data discovery...")
        data_files = engine.discover_data_files()
        print(f"✓ Found {len(data_files)} data files")
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting minimal test...")
    success = test_minimal()
    
    if success:
        print("✓ Minimal test completed!")
    else:
        print("✗ Minimal test failed!")
