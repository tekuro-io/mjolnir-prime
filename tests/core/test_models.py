# ===== FILE: tests/core/test_models.py =====
"""
Tests for core models.
"""

import pytest
from datetime import datetime
from trading_simulator.core.models import CandlestickTick, Order, Trade
from trading_simulator.core.types import CandleType, OrderType, OrderSide


class TestCandlestickTick:
    def test_basic_candlestick_creation(self):
        """Test basic candlestick creation"""
        tick = CandlestickTick(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=10000
        )
        
        assert tick.symbol == 'AAPL'
        assert tick.open == 100.0
        assert tick.candle_type == CandleType.BULLISH
    
    def test_candle_classification_bullish(self):
        """Test bullish candle classification"""
        tick = CandlestickTick('AAPL', datetime.now(), 100.0, 102.0, 99.0, 101.0, 10000)
        assert tick.candle_type == CandleType.BULLISH
    
    def test_candle_classification_bearish(self):
        """Test bearish candle classification"""
        tick = CandlestickTick('AAPL', datetime.now(), 101.0, 102.0, 99.0, 100.0, 10000)
        assert tick.candle_type == CandleType.BEARISH
    
    def test_candle_classification_doji(self):
        """Test doji candle classification"""
        tick = CandlestickTick('AAPL', datetime.now(), 100.0, 101.0, 99.0, 100.001, 10000)
        assert tick.candle_type == CandleType.DOJI
    
    def test_candle_properties(self):
        """Test candlestick properties"""
        tick = CandlestickTick('AAPL', datetime.now(), 100.0, 103.0, 98.0, 102.0, 10000)
        
        assert tick.body_size == 2.0
        assert tick.upper_shadow == 1.0  # 103 - 102
        assert tick.lower_shadow == 2.0  # 100 - 98
        assert tick.total_range == 5.0   # 103 - 98


class TestOrder:
    def test_order_creation(self):
        """Test order creation"""
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert not order.filled
        assert order.id is not None
    
    def test_limit_order(self):
        """Test limit order creation"""
        order = Order(
            symbol='GOOGL',
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=2800.0
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.price == 2800.0