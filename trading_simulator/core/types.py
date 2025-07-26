"""
Core type definitions for the trading simulator.
Contains all enums and basic type definitions.
"""

from enum import Enum


class OrderType(Enum):
    """Types of orders that can be placed"""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Side of the order - buy or sell"""
    BUY = "buy"
    SELL = "sell"


class CandleType(Enum):
    """Types of candlestick patterns"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    NEUTRAL = "neutral"


class PatternType(Enum):
    """Types of technical analysis patterns"""
    FLAT_TOP_BREAKOUT = "flat_top_breakout"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"