# ===== FILE: trading_simulator/core/__init__.py =====
"""Core components of the trading simulator."""

from .types import OrderType, OrderSide, CandleType, PatternType
from .models import CandlestickTick, Order, Trade, PatternMatch
from .exceptions import (
    TradingSimulatorError, InsufficientFundsError, 
    InsufficientSharesError, InvalidOrderError
)

__all__ = [
    'OrderType', 'OrderSide', 'CandleType', 'PatternType',
    'CandlestickTick', 'Order', 'Trade', 'PatternMatch',
    'TradingSimulatorError', 'InsufficientFundsError', 
    'InsufficientSharesError', 'InvalidOrderError'
]
