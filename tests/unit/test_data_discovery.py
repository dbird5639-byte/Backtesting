#!/usr/bin/env python3
"""
Test data file discovery
"""

import sys
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_data_discovery():
    """Test data file discovery"""
    try:
        from engines.basic_engine import BasicEngine
        
        # Create engine
        engine = BasicEngine()
        
        # Discover data files
        data_files = engine.discover_data_files()
        
        print(f"Discovered {len(data_files)} data files")
        
        # Show first few files
        for i, file_path in enumerate(data_files[:10]):
            print(f"  {i+1}: {file_path}")
        
        if len(data_files) > 10:
            print(f"  ... and {len(data_files) - 10} more files")
        
        return len(data_files) > 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing data file discovery...")
    success = test_data_discovery()
    
    if success:
        print("✓ Data discovery test passed!")
    else:
        print("✗ Data discovery test failed!")
