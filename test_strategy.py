#!/usr/bin/env python3
"""
Test Strategy Class
"""

from backtesting import Strategy
import inspect

# Check Strategy class signature
print("Strategy class signature:")
print(inspect.signature(Strategy.__init__))

# Check if Strategy is abstract
print(f"\nStrategy is abstract: {inspect.isabstract(Strategy)}")

# Try to create a simple strategy
class TestStrategy(Strategy):
    def init(self):
        pass
    
    def next(self):
        pass

print(f"\nTestStrategy created successfully: {TestStrategy}")

# Try to instantiate it
try:
    strategy = TestStrategy()
    print("Strategy instantiated successfully without arguments")
except Exception as e:
    print(f"Error instantiating strategy: {e}")

# Check what arguments Strategy expects
print(f"\nStrategy.__init__ args: {inspect.getfullargspec(Strategy.__init__)}")
