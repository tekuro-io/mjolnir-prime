# ===== FILE: config/trading_config.py =====
"""
Configuration settings for the trading simulator.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PatternDetectorConfig:
    """Configuration for pattern detection"""
    tolerance: float = 0.01
    lookback_window: int = 50
    min_confidence_threshold: float = 0.6


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management"""
    initial_balance: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_portfolio_risk: float = 0.2  # 20% total risk exposure
    commission_per_trade: float = 0.0  # Commission per trade


@dataclass
class TradingEngineConfig:
    """Main trading engine configuration"""
    symbols: List[str]
    portfolio_config: PortfolioConfig
    pattern_config: PatternDetectorConfig
    enable_slippage: bool = False
    slippage_bps: float = 5.0  # Slippage in basis points


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_balance: float = 100000.0
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    benchmark_symbol: str = "SPY"
    calculate_metrics: bool = True


# Predefined configurations
CONSERVATIVE_CONFIG = TradingEngineConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    portfolio_config=PortfolioConfig(
        initial_balance=100000.0,
        max_position_size=0.05,  # 5% per position
        max_portfolio_risk=0.15,  # 15% total risk
    ),
    pattern_config=PatternDetectorConfig(
        tolerance=0.005,
        min_confidence_threshold=0.8,  # High confidence only
    )
)

AGGRESSIVE_CONFIG = TradingEngineConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD'],
    portfolio_config=PortfolioConfig(
        initial_balance=100000.0,
        max_position_size=0.15,  # 15% per position
        max_portfolio_risk=0.4,   # 40% total risk
    ),
    pattern_config=PatternDetectorConfig(
        tolerance=0.02,
        min_confidence_threshold=0.5,  # Lower confidence threshold
    )
)

BALANCED_CONFIG = TradingEngineConfig(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    portfolio_config=PortfolioConfig(
        initial_balance=100000.0,
        max_position_size=0.1,   # 10% per position
        max_portfolio_risk=0.25,  # 25% total risk
    ),
    pattern_config=PatternDetectorConfig(
        tolerance=0.01,
        min_confidence_threshold=0.65,
    )
)


def create_custom_config(symbols: List[str], initial_balance: float = 100000.0,
                        risk_level: str = "balanced") -> TradingEngineConfig:
    """Create a custom configuration based on risk level"""
    
    risk_profiles = {
        "conservative": (0.05, 0.15, 0.8),   # (max_pos, max_risk, confidence)
        "balanced": (0.1, 0.25, 0.65),
        "aggressive": (0.15, 0.4, 0.5),
    }
    
    if risk_level not in risk_profiles:
        risk_level = "balanced"
    
    max_pos, max_risk, confidence = risk_profiles[risk_level]
    
    return TradingEngineConfig(
        symbols=symbols,
        portfolio_config=PortfolioConfig(
            initial_balance=initial_balance,
            max_position_size=max_pos,
            max_portfolio_risk=max_risk,
        ),
        pattern_config=PatternDetectorConfig(
            min_confidence_threshold=confidence
        )
    )
