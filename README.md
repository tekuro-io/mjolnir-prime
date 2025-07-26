# ===== FILE: README.md =====
# Trading Simulator

![Alt text](mjolnir.png)

A modular, extensible stock trading simulation framework built in Python. Designed for backtesting trading algorithms, pattern recognition, and strategy development.

## Features

🚀 **Pattern Detection**: Advanced candlestick pattern recognition with tolerance-based matching  
📊 **Portfolio Management**: Complete portfolio tracking with P&L, risk management, and position sizing  
🤖 **Algorithm Framework**: Extensible base classes for creating custom trading algorithms  
📈 **Backtesting**: Comprehensive backtesting framework with performance metrics  
📁 **Data Loading**: Support for JSON, CSV, and mock data generation  
⚙️ **Modular Design**: Clean separation of concerns for easy extension and testing  

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-simulator.git
cd trading-simulator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from trading_simulator import create_demo_setup

# Create a trading engine with default strategies
engine, factory = create_demo_setup()

# Process some market data
from trading_simulator.data.loaders import MockDataGenerator

# Generate test data
candlesticks = MockDataGenerator.generate_trending_data(
    'AAPL', start_price=150.0, num_candles=100
)

# Process the data
for candle in candlesticks:
    engine.process_tick(candle)

# View results
summary = engine.get_portfolio_summary()
print(f"P&L: ${summary['total_pnl']:,.2f}")
```

### Running Examples

```bash
# Run all demos
python -m trading_simulator.examples.basic_demo

# Run specific demos
python -m trading_simulator.examples.basic_demo basic
python -m trading_simulator.examples.basic_demo algorithms
python -m trading_simulator.examples.basic_demo backtest
```

## Architecture

```
trading_simulator/
├── core/              # Basic types, models, exceptions
├── portfolio/         # Portfolio and position management
├── trading/           # Trading engine and order execution
├── patterns/          # Pattern detection and strategies
├── data/              # Data loading and validation
├── algorithms/        # Algorithm framework and base classes
└── examples/          # Usage examples and demos
```

## Key Components

### Trading Engine
The main orchestrator that processes market data, detects patterns, and executes trades.

### Pattern Detection
Advanced pattern recognition supporting:
- Flat top breakouts
- Bullish/bearish reversals
- Double tops/bottoms
- Custom pattern development

### Portfolio Management
- Real-time P&L tracking
- Position management with average cost basis
- Risk management and position sizing
- Trade history and performance metrics

### Algorithm Framework
- Abstract base classes for different algorithm types
- Pattern-based algorithms
- Momentum and mean reversion strategies
- Backtesting and performance comparison

## Creating Custom Strategies

```python
from trading_simulator.patterns.strategies import PatternStrategies
from trading_simulator.core.types import PatternType, OrderSide

def my_custom_strategy(engine):
    def strategy(pattern):
        if pattern.confidence > 0.8:
            print(f"High confidence pattern: {pattern.pattern_type}")
            engine.place_market_order(pattern.symbol, OrderSide.BUY, 100)
    return strategy

# Register the strategy
engine.register_pattern_strategy(PatternType.FLAT_TOP_BREAKOUT, my_custom_strategy(engine))
```

## Configuration

Use the configuration system for different trading setups:

```python
from trading_simulator.config.trading_config import AGGRESSIVE_CONFIG, create_custom_config

# Use predefined configuration
config = AGGRESSIVE_CONFIG

# Or create custom configuration
config = create_custom_config(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    initial_balance=50000.0,
    risk_level="conservative"
)
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=trading_simulator
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

To kill

## Disclaimer

Yeet