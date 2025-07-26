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
            if position.quantity > 0:
                # Use current market price if available, otherwise use average cost
                current_price = current_prices.get(symbol, position.avg_cost)
                total_value += position.quantity * current_price
        return total_value
    
    def get_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total profit/loss vs initial balance"""
        return self.get_portfolio_value(current_prices) - self.initial_balance
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L for all positions"""
        total_unrealized = 0.0
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = current_prices.get(symbol, position.avg_cost)
                total_unrealized += position.unrealized_pnl(current_price)
        return total_unrealized
    
    def get_realized_pnl(self) -> float:
        """Calculate realized P&L from completed trades"""
        # Track cost basis properly for realized gains/losses
        symbol_cost_basis = {}
        realized_pnl = 0.0
        
        for trade in self.trades:
            symbol = trade.symbol
            
            if symbol not in symbol_cost_basis:
                symbol_cost_basis[symbol] = {'shares': 0, 'total_cost': 0.0}
            
            if trade.side.value == 'buy':  # Fix enum comparison
                # Add to cost basis
                symbol_cost_basis[symbol]['shares'] += trade.quantity
                symbol_cost_basis[symbol]['total_cost'] += trade.quantity * trade.price
            else:  # sell
                # Calculate realized gain/loss
                if symbol_cost_basis[symbol]['shares'] > 0:
                    avg_cost = symbol_cost_basis[symbol]['total_cost'] / symbol_cost_basis[symbol]['shares']
                    realized_pnl += (trade.price - avg_cost) * trade.quantity
                    
                    # Reduce cost basis
                    symbol_cost_basis[symbol]['shares'] -= trade.quantity
                    symbol_cost_basis[symbol]['total_cost'] -= avg_cost * trade.quantity
        
        return realized_pnl
    
    def get_positions_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get summary of all positions"""
        summary = {}
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = current_prices.get(symbol, position.avg_cost)
                market_value = position.quantity * current_price
                unrealized_pnl = position.unrealized_pnl(current_price)
                pnl_percent = (unrealized_pnl / (position.quantity * position.avg_cost) * 100) if position.avg_cost > 0 else 0
                
                summary[symbol] = {
                    'quantity': position.quantity,
                    'avg_cost': position.avg_cost,
                    'current_price': current_price,
                    'market_value': market_value,
                    'cost_basis': position.quantity * position.avg_cost,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percent': pnl_percent
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