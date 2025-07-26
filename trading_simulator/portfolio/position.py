"""
Position management for individual stock holdings.
"""

from dataclasses import dataclass
from ..core.exceptions import InsufficientSharesError


@dataclass
class Position:
    """Represents a position in a single stock"""
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value using average cost (for unrealized P&L calculation)"""
        return self.quantity * self.avg_cost
    
    def current_market_value(self, current_price: float) -> float:
        """Market value at current price"""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized profit/loss at current price"""
        return (current_price - self.avg_cost) * self.quantity
    
    def add_shares(self, quantity: int, price: float) -> None:
        """Add shares to position, updating average cost"""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.quantity == 0:
            self.avg_cost = price
        else:
            total_cost = (self.quantity * self.avg_cost) + (quantity * price)
            self.quantity += quantity
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0.0
        
    def remove_shares(self, quantity: int) -> bool:
        """Remove shares from position. Returns True if successful"""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if quantity > self.quantity:
            return False
            
        self.quantity -= quantity
        if self.quantity == 0:
            self.avg_cost = 0.0
        return True
    
    def can_sell(self, quantity: int) -> bool:
        """Check if we can sell the requested quantity"""
        return self.quantity >= quantity and quantity > 0
    
    def __str__(self) -> str:
        return f"Position({self.symbol}: {self.quantity} @ ${self.avg_cost:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()