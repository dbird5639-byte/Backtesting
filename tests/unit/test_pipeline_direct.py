#!/usr/bin/env python3
"""
Direct pipeline test to bypass hanging imports
"""

import sys
import os
from pathlib import Path

# Add the Engines directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'Engines'))

def test_pipeline_direct():
    """Test pipeline execution directly"""
    try:
        print("Testing direct pipeline execution...")
        
        # Try to import and run the basic engine directly
        print("1. Importing basic engine...")
        from Engines.engines.basic_engine import BasicEngine
        print("✓ Basic engine imported")
        
        print("2. Creating engine instance...")
        engine = BasicEngine()
        print("✓ Engine created")
        
        print("3. Discovering strategies...")
        strategies = engine.discover_strategies()
        print(f"✓ Found {len(strategies)} strategies")
        
        print("4. Discovering data files...")
        data_files = engine.discover_data_files()
        print(f"✓ Found {len(data_files)} data files")
        
        if strategies and data_files:
            print("5. Running pipeline...")
            results = engine.run()
            print(f"✓ Pipeline completed with {len(results)} results")
            return True
        else:
            print("✗ No strategies or data files found")
            return False
            
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_direct()
    
    if success:
        print("✓ Pipeline test passed!")
    else:
        print("✗ Pipeline test failed!")
