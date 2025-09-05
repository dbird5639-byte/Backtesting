#!/usr/bin/env python3
"""
Update Results Path Script
Updates all references from Engines/Results to Results in the codebase
"""

import os
import re
from pathlib import Path

def update_file_paths(file_path):
    """Update file paths in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace various patterns
        patterns = [
            (r'Engines/Results', 'Results'),
            (r'Engines\\Results', 'Results'),
            (r'Engines\\\\Results', 'Results'),
            (r'"Engines/Results"', '"Results"'),
            (r"'Engines/Results'", "'Results'"),
            (r'Path\(.*Engines.*Results.*\)', 'Path("Results")'),
        ]
        
        original_content = content
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
            return True
        else:
            print(f"⏭️  No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all files"""
    print("🔄 Updating Results paths from Engines/Results to Results...")
    
    # Files to update
    files_to_update = [
        "simple_core_engine.py",
        "run_all_engines_in_order.py", 
        "run_all_9_engines.py",
        "run_real_data_engines.py",
        "run_core_engine_only.py",
        "comprehensive_engine_runner.py",
        "simple_engine_runner.py",
        "recursive_engine_runner.py",
        "simple_recursive_runner.py",
        "Config/settings.py",
        "Engines/core_engine.py",
        "Engines/engine_factory.py",
        "Engines/enhanced_risk_engine.py",
        "Engines/regime_detection_engine.py",
        "Engines/regime_overlay_engine.py",
        "Engines/enhanced_visualization_engine.py",
        "Engines/fibonacci_gann_engine.py",
        "Engines/performance_engine.py",
        "Engines/validation_engine.py",
        "Engines/portfolio_engine.py",
        "Engines/risk_engine.py",
        "Engines/ml_engine.py",
    ]
    
    updated_count = 0
    total_files = len(files_to_update)
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            if update_file_paths(file_path):
                updated_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n✅ Update complete! {updated_count}/{total_files} files updated")
    print("📁 Results will now be saved to: Results/")

if __name__ == "__main__":
    main()
