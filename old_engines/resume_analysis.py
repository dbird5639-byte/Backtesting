#!/usr/bin/env python3
"""
Resume Analysis Script

This script analyzes existing results and helps resume backtesting from where it left off.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path so we can import the engines
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from base_engine import BaseEngine, EngineConfig

def analyze_existing_results():
    """Analyze existing results to understand current state"""
    results_base_dir = r"C:\Users\andre\OneDrive\Desktop\MasterCode\Backtesting\results"
    
    print("🔍 Analyzing Existing Results")
    print("=" * 50)
    
    if not os.path.exists(results_base_dir):
        print("❌ Results directory does not exist")
        return
    
    # Find all engine result directories
    engine_dirs = []
    for item in os.listdir(results_base_dir):
        item_path = os.path.join(results_base_dir, item)
        if os.path.isdir(item_path) and not item.startswith('logs'):
            engine_dirs.append(item)
    
    if not engine_dirs:
        print("❌ No engine result directories found")
        return
    
    print(f"📁 Found {len(engine_dirs)} engine result directories:")
    
    total_results = 0
    engine_summaries = {}
    
    for engine_dir in sorted(engine_dirs):
        engine_path = os.path.join(results_base_dir, engine_dir)
        
        # Count strategies and results
        strategies = []
        results_count = 0
        
        for strategy_dir in os.listdir(engine_path):
            strategy_path = os.path.join(engine_path, strategy_dir)
            if os.path.isdir(strategy_path):
                strategies.append(strategy_dir)
                # Count JSON files (results)
                json_files = [f for f in os.listdir(strategy_path) if f.endswith('.json')]
                results_count += len(json_files)
        
        total_results += results_count
        
        engine_summaries[engine_dir] = {
            'strategies': strategies,
            'results_count': results_count,
            'path': engine_path
        }
        
        print(f"  📊 {engine_dir}:")
        print(f"     - Strategies: {len(strategies)}")
        print(f"     - Results: {results_count}")
        if strategies:
            print(f"     - Strategy list: {', '.join(strategies[:3])}{'...' if len(strategies) > 3 else ''}")
    
    print(f"\n📈 Total Results: {total_results}")
    
    # Analyze specific engines
    print("\n🔍 Detailed Analysis:")
    
    for engine_dir, summary in engine_summaries.items():
        print(f"\n📊 {engine_dir}:")
        
        if summary['results_count'] > 0:
            # Find the most recent result
            latest_time = 0
            latest_file = None
            
            for strategy in summary['strategies']:
                strategy_path = os.path.join(summary['path'], strategy)
                for file in os.listdir(strategy_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(strategy_path, file)
                        file_time = os.path.getmtime(file_path)
                        if file_time > latest_time:
                            latest_time = file_time
                            latest_file = f"{strategy}/{file.replace('.json', '')}"
            
            if latest_file:
                latest_date = datetime.fromtimestamp(latest_time)
                print(f"  ✅ Last processed: {latest_file}")
                print(f"  ⏰ Last modified: {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show sample results
            print(f"  📄 Sample results:")
            for strategy in summary['strategies'][:2]:  # Show first 2 strategies
                strategy_path = os.path.join(summary['path'], strategy)
                json_files = [f for f in os.listdir(strategy_path) if f.endswith('.json')]
                if json_files:
                    print(f"    - {strategy}: {len(json_files)} files")
                    # Show a few sample files
                    sample_files = json_files[:3]
                    for file in sample_files:
                        data_name = file.replace('.json', '')
                        print(f"      • {data_name}")
        else:
            print(f"  ⚠️  No results found")
    
    return engine_summaries

def get_resume_recommendations(engine_summaries):
    """Get recommendations for resuming backtesting"""
    print("\n💡 Resume Recommendations:")
    print("=" * 50)
    
    recommendations = []
    
    for engine_dir, summary in engine_summaries.items():
        if summary['results_count'] > 0:
            # Engine has results - can resume
            recommendations.append({
                'engine': engine_dir,
                'action': 'resume',
                'results_count': summary['results_count'],
                'strategies_count': len(summary['strategies'])
            })
            print(f"✅ {engine_dir}: Can resume (has {summary['results_count']} results)")
        else:
            # Engine has no results - start fresh
            recommendations.append({
                'engine': engine_dir,
                'action': 'start_fresh',
                'results_count': 0,
                'strategies_count': 0
            })
            print(f"🆕 {engine_dir}: Start fresh (no existing results)")
    
    return recommendations

def create_resume_script(recommendations):
    """Create a script to resume backtesting"""
    print("\n📝 Resume Script:")
    print("=" * 50)
    
    script_content = """#!/usr/bin/env python3
# Resume Backtesting Script
# Generated on {}

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from run_engines import *

def resume_backtesting():
    \"\"\"Resume backtesting from where it left off\"\"\"
    print("🔄 Resuming Backtesting...")
    print("=" * 40)
    
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    for rec in recommendations:
        if rec['action'] == 'resume':
            script_content += f"""
    # Resume {rec['engine']}
    print("\\n🚀 Resuming {rec['engine']}...")
    try:
        run_{rec['engine'].split('_')[0]}_engine()
        print("✅ {rec['engine']} completed successfully")
    except Exception as e:
        print(f"❌ Error resuming {rec['engine']}: {{e}}")
"""
        else:
            script_content += f"""
    # Start fresh {rec['engine']}
    print("\\n🆕 Starting fresh {rec['engine']}...")
    try:
        run_{rec['engine'].split('_')[0]}_engine()
        print("✅ {rec['engine']} completed successfully")
    except Exception as e:
        print(f"❌ Error starting {rec['engine']}: {{e}}")
"""
    
    script_content += """
if __name__ == "__main__":
    resume_backtesting()
"""
    
    # Save the script
    script_path = os.path.join(current_dir, "resume_backtesting.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"📄 Resume script created: {script_path}")
    print("\nTo run the resume script:")
    print(f"  python {script_path}")
    
    return script_path

def main():
    """Main function"""
    print("🔄 Backtesting Resume Analysis")
    print("=" * 50)
    
    # Analyze existing results
    engine_summaries = analyze_existing_results()
    
    if not engine_summaries:
        print("\n❌ No results to analyze")
        return
    
    # Get recommendations
    recommendations = get_resume_recommendations(engine_summaries)
    
    # Create resume script
    script_path = create_resume_script(recommendations)
    
    print(f"\n🎉 Resume analysis completed!")
    print(f"📄 Resume script: {script_path}")
    print("\nNext steps:")
    print("1. Review the analysis above")
    print("2. Run the resume script to continue backtesting")
    print("3. Or use run_engines.py for interactive selection")

if __name__ == "__main__":
    main() 