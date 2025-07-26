"""
Main trading engine that orchestrates all trading operations.
"""

from typing import List, Dict, Callable, Optional
from datetime import datetime

from ..portfolio.portfolio import Portfolio
from ..patterns.detector import PatternDetector
from ..core.models import CandlestickTick, Order, Trade, PatternMatch
from ..core.types import OrderSide, OrderType, PatternType
from ..core.exceptions import InvalidOrderError


class TradeEngine:
    """Main trading engine that processes ticks and executes trades"""
    
    def __init__(self, portfolio: Portfolio, pattern_detector: PatternDetector, symbols: List[str]):
        self.portfolio = portfolio
        self.symbols = symbols
        self.current_prices: Dict[str, float] = {}
        self.pending_orders: List[Order] = []
        self.executed_orders: List[Order] = []
        
        self.pattern_detector = pattern_detector
        self.detected_patterns: List[PatternMatch] = []
        self.algorithm_callbacks: List[Callable[[CandlestickTick, List[PatternMatch]], None]] = []
        
    def process_tick(self, tick: CandlestickTick) -> None:
        """Process incoming candlestick tick"""
        if tick.symbol not in self.symbols:
            raise InvalidOrderError(f"Symbol {tick.symbol} not in allowed symbols")
            
        self.current_prices[tick.symbol] = tick.close
        
        # Detect patterns
        new_patterns = self.pattern_detector.add_candle(tick)
        self.detected_patterns.extend(new_patterns)
        
        # Process pending orders
        self._process_pending_orders(tick)
        
        # Notify algorithm callbacks
        for callback in self.algorithm_callbacks:
            try:
                callback(tick, new_patterns)
            except Exception as e:
                print(f"Error in algorithm callback: {e}")
    
    def register_algorithm(self, callback: Callable[[CandlestickTick, List[PatternMatch]], None]):
        """Register algorithm callback for tick processing"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.algorithm_callbacks.append(callback)
    
    def register_pattern_strategy(self, pattern_type: PatternType, 
                                strategy: Callable[[PatternMatch], None]):
        """Register strategy for specific pattern type"""
        self.pattern_detector.register_pattern_callback(pattern_type, strategy)
    
    def _process_pending_orders(self, tick: CandlestickTick) -> None:
        """Process any pending limit orders that can be filled"""
        orders_to_remove = []
        
        for order in self.pending_orders:
            if order.symbol != tick.symbol:
                continue
                
            if self._can_fill_order(order, tick):
                fill_price = self._get_fill_price(order, tick)
                if self._execute_order(order, fill_price, tick.timestamp):
                    order.filled = True
                    order.fill_price = fill_price
                    order.fill_timestamp = tick.timestamp
                    self.executed_orders.append(order)
                    orders_to_remove.append(order)
        
        for order in orders_to_remove:
            self.pending_orders.remove(order)
    
    def _can_fill_order(self, order: Order, tick: CandlestickTick) -> bool:
        """Check if order can be filled based on tick data"""
        if order.order_type == OrderType.MARKET:
            return True
        elif order.order_type == OrderType.LIMIT and order.price:
            if order.side == OrderSide.BUY:
                return tick.low <= order.price
            else:  # SELL
                return tick.high >= order.price
        return False
    
    def _get_fill_price(self, order: Order, tick: CandlestickTick) -> float:
        """Determine fill price for order"""
        if order.order_type == OrderType.MARKET:
            return tick.close  # Simplified - could use open for more realism
        else:
            return order.price  # Limit order fills at limit price
    
    def _execute_order(self, order: Order, price: float, timestamp: datetime) -> bool:
        """Execute the order through portfolio"""
        try:
            if order.side == OrderSide.BUY:
                success = self.portfolio.execute_buy(order.symbol, order.quantity, price)
            else:
                success = self.portfolio.execute_sell(order.symbol, order.quantity, price)
            
            if success:
                trade = Trade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=price,
                    timestamp=timestamp,
                    order_id=order.id
                )
                self.portfolio.add_trade(trade)
            
            return success
        except Exception as e:
            print(f"Failed to execute order {order.id}: {e}")
            return False
    
    def place_market_order(self, symbol: str, side: OrderSide, quantity: int) -> str:
        """Place a market order"""
        if symbol not in self.symbols:
            raise InvalidOrderError(f"Symbol {symbol} not allowed")
        if quantity <= 0:
            raise InvalidOrderError("Quantity must be positive")
            
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        # Market orders execute immediately if possible
        if symbol in self.current_prices:
            price = self.current_prices[symbol]
            if self._execute_order(order, price, datetime.now()):
                order.filled = True
                order.fill_price = price
                order.fill_timestamp = datetime.now()
                self.executed_orders.append(order)
                return order.id
        
        # If we can't execute immediately, add to pending
        self.pending_orders.append(order)
        return order.id
    
    def place_limit_order(self, symbol: str, side: OrderSide, quantity: int, price: float) -> str:
        """Place a limit order"""
        if symbol not in self.symbols:
            raise InvalidOrderError(f"Symbol {symbol} not allowed")
        if quantity <= 0:
            raise InvalidOrderError("Quantity must be positive")
        if price <= 0:
            raise InvalidOrderError("Price must be positive")
            
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        
        self.pending_orders.append(order)
        return order.id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        for order in self.pending_orders:
            if order.id == order_id:
                self.pending_orders.remove(order)
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of an order"""
        # Check executed orders first
        for order in self.executed_orders:
            if order.id == order_id:
                return order
        
        # Check pending orders
        for order in self.pending_orders:
            if order.id == order_id:
                return order
        
        return None
    
    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary"""
        return {
            'cash': self.portfolio.cash,
            'initial_balance': self.portfolio.initial_balance,
            'positions': self.portfolio.get_positions_summary(self.current_prices),
            'portfolio_value': self.portfolio.get_portfolio_value(self.current_prices),
            'total_pnl': self.portfolio.get_pnl(self.current_prices),
            'unrealized_pnl': self.portfolio.get_unrealized_pnl(self.current_prices),
            'realized_pnl': self.portfolio.get_realized_pnl(),
            'current_prices': self.current_prices.copy(),
            'patterns_detected': len(self.detected_patterns),
            'pending_orders': len(self.pending_orders),
            'executed_orders': len(self.executed_orders),
            'recent_patterns': [
                {
                    'type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'symbol': p.symbol,
                    'timestamp': p.timestamp.isoformat(),
                    'trigger_price': p.trigger_price
                } for p in self.detected_patterns[-5:]  # Last 5 patterns
            ]
        }
    
    def get_trading_statistics(self) -> dict:
        """Get trading performance statistics"""
        if not self.portfolio.trades:
            return {'total_trades': 0}
        
        buy_trades = [t for t in self.portfolio.trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in self.portfolio.trades if t.side == OrderSide.SELL]
        
        return {
            'total_trades': len(self.portfolio.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': sum(t.quantity for t in self.portfolio.trades),
            'avg_trade_size': sum(t.quantity for t in self.portfolio.trades) / len(self.portfolio.trades),
            'total_traded_value': sum(t.quantity * t.price for t in self.portfolio.trades),
            'patterns_traded_on': len([p for p in self.detected_patterns if p.confidence > 0.6])
        }
    
    def reset(self):
        """Reset the engine state (useful for backtesting)"""
        self.current_prices.clear()
        self.pending_orders.clear()
        self.executed_orders.clear()
        self.detected_patterns.clear()
        self.pattern_detector.clear_history()
    
    def __str__(self) -> str:
        return f"TradeEngine(Symbols: {len(self.symbols)}, Patterns: {len(self.detected_patterns)}, Orders: {len(self.executed_orders)})"
    
    def __repr__(self) -> str:
        return self.__str__()