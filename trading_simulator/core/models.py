"""
Core data models for the trading simulator.
Contains all dataclasses and model definitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from .types import OrderType, OrderSide, CandleType, PatternType


@dataclass
class CandlestickTick:
    """Minute candlestick data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    candle_type: Optional[CandleType] = None
    
    def __post_init__(self):
        if self.candle_type is None:
            self.candle_type = self.classify_candle()
    
    def classify_candle(self, tolerance: float = 0.005) -> CandleType:
        """Classify candlestick type with tolerance for neutral/doji detection"""
        body = abs(self.close - self.open)
        total_range = self.high - self.low
        
        if total_range == 0:
            return CandleType.DOJI
            
        body_ratio = body / total_range
        upper_shadow = self.high - max(self.open, self.close)
        lower_shadow = min(self.open, self.close) - self.low
        
        if body < tolerance or body_ratio < 0.1:
            return CandleType.DOJI
        
        if lower_shadow > 2 * body and upper_shadow <= body * 0.5:
            return CandleType.HAMMER if self.close > self.open else CandleType.HANGING_MAN
        
        if upper_shadow > 2 * body and lower_shadow <= body * 0.5:
            return CandleType.SHOOTING_STAR
        
        return CandleType.BULLISH if self.close > self.open else CandleType.BEARISH
    
    @property
    def body_size(self) -> float:
        """Size of the candlestick body"""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Length of upper shadow"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Length of lower shadow"""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Total price range of the candle"""
        return self.high - self.low


@dataclass
class Order:
    """Represents a trading order"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    filled: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None


@dataclass
class Trade:
    """Represents an executed trade"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: str = ""


@dataclass
class PatternMatch:
    """Represents a detected pattern"""
    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    symbol: str
    trigger_price: float
    candles_involved: List[CandlestickTick]
    metadata: Dict = field(default_factory=dict)