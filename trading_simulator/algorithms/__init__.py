# ===== FILE: trading_simulator/algorithms/__init__.py =====
"""Trading algorithm framework."""

from .base import (
    TradingAlgorithm, PatternBasedAlgorithm, MomentumAlgorithm, 
    MeanReversionAlgorithm, SimpleBreakoutAlgorithm, SimpleMomentumAlgorithm,
    AlgorithmManager, BacktestRunner
)

__all__ = [
    'TradingAlgorithm', 'PatternBasedAlgorithm', 'MomentumAlgorithm',
    'MeanReversionAlgorithm', 'SimpleBreakoutAlgorithm', 'SimpleMomentumAlgorithm',
    'AlgorithmManager', 'BacktestRunner'
]