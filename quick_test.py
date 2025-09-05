#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "Engines"))

from Engines.core_engine import CoreEngine, EngineConfig

print("Starting Core Engine...")
config = EngineConfig()
config.parallel_workers = 1
config.verbose = True
config.results_path = "Results"

engine = CoreEngine(config)
print("Engine created, running...")
engine.run()
print("Engine completed!")
