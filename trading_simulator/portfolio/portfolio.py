"""
Portfolio management for cash and positions.
"""

from typing import Dict, List
from .position import Position
from ..core.models import Trade
from ..core.types import OrderSide
from ..core.exceptions import InsufficientFundsError, InsufficientSharesError


class Portfolio:
    """Manages cash balance and stock positions"""
    
    def __init__(self, initial_balance: float):
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
            
        self.cash = initial_balance
        self.initial_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol, creating if doesn't exist"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def can_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if we have enough cash to buy"""
        if quantity <= 0 or price <= 0:
            return False
        cost = quantity * price
        return self.cash >= cost
    
    def can_sell(self, symbol: str, quantity: int) -> bool:
        """Check if we have enough shares to sell"""
        if quantity <= 0:
            return False
        position = self.get_position(symbol)
        return position.can_sell(quantity)
    
    def execute_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a buy order"""
        if not self.can_buy(symbol, quantity, price):
            cost = quantity * price
            raise InsufficientFundsError(cost, self.cash)
        
        cost = quantity * price
        self.cash -= cost
        position = self.get_position(symbol)
        position.add_shares(quantity, price)
        return True
    
    def execute_sell(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a sell order"""
        position = self.get_position(symbol)
        
        if not self.can_sell(symbol, quantity):
            raise InsufficientSharesError(symbol, quantity, position.quantity)
        
        if position.remove_shares(quantity):
            proceeds = quantity * price
            self.cash += proceeds
            return True
        return False
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value using current market prices"""
        total_value = self.cash
        for symbol, position in self.positions.items():
            if position.quantity > 0 and symbol in current_prices:
                total_value += position.current_market_value(current_prices[symbol])
        return total_value
    
    def get_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate profit/loss vs initial balance"""
        return self.get_portfolio_value(current_prices) - self.initial_balance
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L for all positions"""
        total_unrealized = 0.0
        for symbol, position in self.positions.items():
            if position.quantity > 0 and symbol in current_prices:
                total_unrealized += position.unrealized_pnl(current_prices[symbol])
        return total_unrealized
    
    def get_realized_pnl(self) -> float:
        """Calculate realized P&L from completed trades"""
        # This is a simplified version - in practice, you'd track cost basis properly
        buy_trades = [t for t in self.trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in self.trades if t.side == OrderSide.SELL]
        
        total_bought = sum(t.quantity * t.price for t in buy_trades)
        total_sold = sum(t.quantity * t.price for t in sell_trades)
        
        return total_sold - total_bought
    
    def get_positions_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get summary of all positions"""
        summary = {}
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = current_prices.get(symbol, position.avg_cost)
                summary[symbol] = {
                    'quantity': position.quantity,
                    'avg_cost': position.avg_cost,
                    'current_price': current_price,
                    'market_value': position.current_market_value(current_price),
                    'unrealized_pnl': position.unrealized_pnl(current_price),
                    'pnl_percent': ((current_price - position.avg_cost) / position.avg_cost * 100) if position.avg_cost > 0 else 0
                }
        return summary
    
    def add_trade(self, trade: Trade) -> None:
        """Add a completed trade to the portfolio history"""
        self.trades.append(trade)
    
    def __str__(self) -> str:
        active_positions = len([p for p in self.positions.values() if p.quantity > 0])
        return f"Portfolio(Cash: ${self.cash:.2f}, Positions: {active_positions}, Trades: {len(self.trades)})"
    
    def __repr__(self) -> str:
        return self.__str__()