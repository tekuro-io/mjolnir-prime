"""
Custom exceptions for the trading simulator.
"""


class TradingSimulatorError(Exception):
    """Base exception for trading simulator"""
    pass


class InsufficientFundsError(TradingSimulatorError):
    """Raised when attempting to buy with insufficient funds"""
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(f"Insufficient funds: need ${required:.2f}, have ${available:.2f}")


class InsufficientSharesError(TradingSimulatorError):
    """Raised when attempting to sell more shares than available"""
    def __init__(self, symbol: str, requested: int, available: int):
        self.symbol = symbol
        self.requested = requested
        self.available = available
        super().__init__(f"Insufficient shares of {symbol}: need {requested}, have {available}")


class InvalidOrderError(TradingSimulatorError):
    """Raised for invalid order parameters"""
    pass


class PatternDetectionError(TradingSimulatorError):
    """Raised when pattern detection encounters an error"""
    pass


class DataLoadingError(TradingSimulatorError):
    """Raised when data loading fails"""
    pass