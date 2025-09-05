# Backtesting Framework

A clean, organized backtesting framework with standalone engines that can be run independently using the Cursor IDE play button.

## 🚀 Quick Start

### Standalone Engines
All engines are now standalone and can be run directly:

1. **Simple Engine**: `python Engines/standalone_simple_engine.py`
2. **Elliott Wave Engine**: `python Engines/standalone_elliott_wave_engine.py`
3. **Statistical Engine**: `python Engines/standalone_statistical_engine.py`
4. **Portfolio Engine**: `python Engines/standalone_portfolio_engine.py`

### Using Cursor IDE
Simply open any standalone engine file and click the play button (▶️) to run it directly.

## 📁 Clean Structure

```
Backtesting/
├── Engines/                          # Core backtesting engines
│   ├── standalone_*.py              # Standalone engines (run directly)
│   ├── strategy_factory.py          # Strategy creation and management
│   ├── comprehensive_*.py           # Advanced validation frameworks
│   └── STANDALONE_ENGINES_README.md # Detailed usage guide
├── Strategies/                       # Trading strategies
├── fetched_data/                     # Market data (CSV files)
├── storage/                          # Results and outputs
├── tests/                           # Organized test files
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   └── performance/                 # Performance tests
├── Config/                          # Configuration files
├── Tool_box/                        # Trading tools and utilities
├── AI-Projects/                     # AI-related projects
└── utils/                          # Utility functions
```

## 🛠️ Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**
   ```bash
   python -c "import pandas, numpy, scipy, sklearn, plotly; print('All dependencies installed!')"
   ```

## 🎯 Key Features

### Standalone Execution
- **No Pipeline Dependencies**: Each engine runs independently
- **Direct Execution**: Use Cursor IDE play button or `python filename.py`
- **Self-Contained**: All necessary imports and configurations included

### Core Engines
- **Simple Engine**: Basic backtesting with fundamental metrics
- **Elliott Wave Engine**: Advanced technical analysis with wave patterns
- **Statistical Engine**: Comprehensive statistical analysis and validation
- **Portfolio Engine**: Multi-strategy portfolio management

### Advanced Validation
- **Comprehensive Validation Framework**: Walk-forward analysis, bootstrap testing
- **Strategy Factory**: Dynamic strategy creation and management
- **Risk Management**: Enhanced risk controls and position sizing

## 📊 Usage Examples

### Running a Simple Backtest
```python
# Open Engines/standalone_simple_engine.py
# Click the play button (▶️) in Cursor IDE
# Or run: python Engines/standalone_simple_engine.py
```

### Running Elliott Wave Analysis
```python
# Open Engines/standalone_elliott_wave_engine.py
# Click the play button (▶️) in Cursor IDE
# Or run: python Engines/standalone_elliott_wave_engine.py
```

### Running Statistical Analysis
```python
# Open Engines/standalone_statistical_engine.py
# Click the play button (▶️) in Cursor IDE
# Or run: python Engines/standalone_statistical_engine.py
```

## 📈 Data Structure

### Market Data
- **Location**: `fetched_data/` directory
- **Format**: CSV files with OHLCV data
- **Symbols**: BTC, ETH, XRP, and other cryptocurrencies
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d

### Results Storage
- **Location**: `storage/` directory
- **Formats**: JSON, CSV, HTML reports
- **Organization**: Timestamped folders for each run

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component testing
- **Performance Tests**: Speed and memory testing

## 🔧 Configuration

### Engine Configuration
Each standalone engine includes default configurations that can be modified:

```python
# Example: Modify SimpleEngineConfig in standalone_simple_engine.py
config = SimpleEngineConfig(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005
)
```

### Strategy Configuration
Strategies can be customized through the strategy factory:

```python
# Example: Create custom strategy
strategy = MomentumStrategy(
    name="Custom_Momentum",
    config={'short_window': 10, 'long_window': 30}
)
```

## 📚 Documentation

- **Standalone Engines Guide**: `Engines/STANDALONE_ENGINES_README.md`
- **Strategy Documentation**: `Strategies/` directory
- **Tool Documentation**: `Tool_box/README.md`

## 🚀 Benefits of Clean Structure

✅ **Easy Navigation**: Clear, organized folder structure  
✅ **Standalone Execution**: Run engines directly with play button  
✅ **No Dependencies**: Each engine is self-contained  
✅ **Clean Codebase**: Removed redundant and outdated files  
✅ **Focused Development**: Work with current, working versions  
✅ **Better Maintenance**: Single source of truth for each component  

## 🔮 Future Enhancements

- **Real-time Data Integration**: Live market data feeds
- **Web Interface**: Interactive analysis dashboard
- **API Integration**: RESTful API for external access
- **Machine Learning**: AI-powered strategy generation
- **Cloud Deployment**: AWS/Google Cloud integration

## 🆘 Support

For questions and support:
1. Check the `STANDALONE_ENGINES_README.md` for detailed usage
2. Review example scripts in the `Engines/` directory
3. Check the logs for detailed error information
4. Open an issue for bugs or feature requests

---

**Note**: This system is designed for research and educational purposes. Always validate results and consider transaction costs, slippage, and market impact in real trading scenarios.