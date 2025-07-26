# ===== FILE: tests/conftest.py =====
"""
Pytest configuration and shared fixtures.
"""

import pytest
from datetime import datetime, timedelta
from trading_simulator.portfolio.portfolio import Portfolio
from trading_simulator.patterns.detector import PatternDetector
from trading_simulator.trading.engine import TradeEngine
from trading_simulator.core.models import CandlestickTick
from trading_simulator.core.types import CandleType


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing"""
    return Portfolio(initial_balance=100000.0)


@pytest.fixture
def sample_pattern_detector():
    """Create a sample pattern detector for testing"""
    return PatternDetector(tolerance=0.01, lookback_window=20)


@pytest.fixture
def sample_trading_engine(sample_portfolio, sample_pattern_detector):
    """Create a sample trading engine for testing"""
    return TradeEngine(sample_portfolio, sample_pattern_detector, ['AAPL', 'GOOGL'])


@pytest.fixture
def sample_candlesticks():
    """Create sample candlestick data for testing"""
    base_time = datetime.now()
    return [
        CandlestickTick('AAPL', base_time, 100.0, 101.0, 99.0, 100.5, 10000),
        CandlestickTick('AAPL', base_time + timedelta(minutes=1), 100.5, 102.0, 100.0, 101.5, 12000),
        CandlestickTick('AAPL', base_time + timedelta(minutes=2), 101.5, 102.5, 101.0, 102.0, 15000),
    ]


@pytest.fixture
def flat_top_pattern_data():
    """Create candlestick data that should trigger a flat top pattern"""
    base_time = datetime.now()
    return [
        CandlestickTick('AAPL', base_time, 100.0, 101.0, 99.5, 100.8, 50000),
        CandlestickTick('AAPL', base_time + timedelta(minutes=1), 100.8, 101.2, 100.0, 100.9, 45000),
        CandlestickTick('AAPL', base_time + timedelta(minutes=2), 100.9, 101.1, 100.1, 100.9, 40000),
        CandlestickTick('AAPL', base_time + timedelta(minutes=3), 100.9, 102.5, 100.8, 102.2, 80000),
    ]

