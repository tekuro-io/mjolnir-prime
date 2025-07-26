"""
Trading Simulator - A modular stock trading simulation framework.

This package provides a comprehensive trading simulator with:
- Pattern detection and technical analysis
- Portfolio management with risk controls
- Algorithm framework with backtesting
- Data loading and validation utilities
- Extensible strategy framework
"""

__version__ = "1.0.0"
__author__ = "Trading Simulator Team"

from .core.types import OrderType, OrderSide, CandleType, PatternType
from .core.models import CandlestickTick, Order, Trade, PatternMatch
from .portfolio.portfolio import Portfolio
from .trading.engine import TradeEngine
from .patterns.detector import PatternDetector
from .patterns.strategies import PatternStrategies, StrategyFactory
from .algorithms.base import TradingAlgorithm, AlgorithmManager, BacktestRunner
from .data.loaders import JsonTickLoader, MockDataGenerator

# Convenience factory functions
def create_engine(initial_balance: float = 100000.0, symbols: list = None, 
                  pattern_tolerance: float = 0.01):
    """Create a basic trading engine with default settings"""
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    portfolio = Portfolio(initial_balance)
    detector = PatternDetector(tolerance=pattern_tolerance)
    return TradeEngine(portfolio, detector, symbols)

def create_demo_setup():
    """Create a demo trading setup with sample strategies"""
    engine = create_engine()
    factory = StrategyFactory(engine)
    factory.create_balanced_setup()
    return engine, factory

__all__ = [
    # Core types
    'OrderType', 'OrderSide', 'CandleType', 'PatternType',
    # Core models
    'CandlestickTick', 'Order', 'Trade', 'PatternMatch',
    # Main components
    'Portfolio', 'TradeEngine', 'PatternDetector',
    # Strategies and algorithms
    'PatternStrategies', 'StrategyFactory', 'TradingAlgorithm', 
    'AlgorithmManager', 'BacktestRunner',
    # Data handling
    'JsonTickLoader', 'MockDataGenerator',
    # Convenience functions
    'create_engine', 'create_demo_setup'
]
