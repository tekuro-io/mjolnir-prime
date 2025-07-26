"""
Order management utilities and validation.
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

from ..core.models import Order, CandlestickTick
from ..core.types import OrderType, OrderSide
from ..core.exceptions import InvalidOrderError


class OrderStatus(Enum):
    """Extended order status tracking"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderValidator:
    """Validate orders before placement"""
    
    @staticmethod
    def validate_order(order: Order, current_price: Optional[float] = None) -> List[str]:
        """
        Validate an order and return list of validation errors
        
        Args:
            order: Order to validate
            current_price: Current market price for additional validation
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Basic validation
        if not order.symbol or not order.symbol.strip():
            errors.append("Symbol cannot be empty")
        
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        
        # Limit order validation
        if order.order_type == OrderType.LIMIT:
            if order.price is None or order.price <= 0:
                errors.append("Limit order must have positive price")
            
            # Check if limit price is reasonable compared to current price
            if current_price and order.price:
                price_diff = abs(order.price - current_price) / current_price
                
                if price_diff > 0.2:  # More than 20% away from current price
                    errors.append(f"Limit price ${order.price:.2f} is far from current price ${current_price:.2f}")
        
        # Market order validation
        elif order.order_type == OrderType.MARKET:
            if order.price is not None:
                errors.append("Market order should not have a price specified")
        
        return errors
    
    @staticmethod
    def is_valid_order(order: Order, current_price: Optional[float] = None) -> bool:
        """Quick validation check"""
        return len(OrderValidator.validate_order(order, current_price)) == 0


class OrderManager:
    """Manage order lifecycle and execution"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_callbacks: Dict[str, List[Callable]] = {}
        self.execution_rules: List[Callable] = []
    
    def add_order(self, order: Order) -> bool:
        """Add an order to management"""
        validation_errors = OrderValidator.validate_order(order)
        if validation_errors:
            raise InvalidOrderError(f"Invalid order: {', '.join(validation_errors)}")
        
        self.orders[order.id] = order
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.filled:
            return False  # Cannot cancel filled order
        
        # Mark as cancelled (you might want to add a cancelled field to Order)
        del self.orders[order_id]
        return True
    
    def get_pending_orders(self, symbol: str = None) -> List[Order]:
        """Get all pending orders, optionally filtered by symbol"""
        pending = [order for order in self.orders.values() if not order.filled]
        
        if symbol:
            pending = [order for order in pending if order.symbol == symbol]
        
        return pending
    
    def get_filled_orders(self, symbol: str = None) -> List[Order]:
        """Get all filled orders, optionally filtered by symbol"""
        filled = [order for order in self.orders.values() if order.filled]
        
        if symbol:
            filled = [order for order in filled if order.symbol == symbol]
        
        return filled
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for order events"""
        if event_type not in self.order_callbacks:
            self.order_callbacks[event_type] = []
        self.order_callbacks[event_type].append(callback)
    
    def _trigger_callback(self, event_type: str, order: Order):
        """Trigger callbacks for order events"""
        if event_type in self.order_callbacks:
            for callback in self.order_callbacks[event_type]:
                try:
                    callback(order)
                except Exception as e:
                    print(f"Error in order callback: {e}")


class SlippageModel:
    """Model for simulating market slippage"""
    
    def __init__(self, fixed_slippage_bps: float = 0, 
                 impact_coefficient: float = 0.001,
                 volatility_factor: float = 0.1):
        """
        Initialize slippage model
        
        Args:
            fixed_slippage_bps: Fixed slippage in basis points (0.01% = 1 bp)
            impact_coefficient: Market impact coefficient
            volatility_factor: How much volatility affects slippage
        """
        self.fixed_slippage_bps = fixed_slippage_bps
        self.impact_coefficient = impact_coefficient
        self.volatility_factor = volatility_factor
    
    def calculate_slippage(self, order: Order, market_price: float, 
                          market_volume: int = 100000, 
                          volatility: float = 0.02) -> float:
        """
        Calculate slippage for an order
        
        Args:
            order: The order being executed
            market_price: Current market price
            market_volume: Current market volume
            volatility: Current market volatility
            
        Returns:
            Slippage as percentage of price (0.01 = 1%)
        """
        # Fixed slippage component
        fixed_slip = self.fixed_slippage_bps / 10000  # Convert bps to percentage
        
        # Market impact based on order size relative to volume
        volume_ratio = order.quantity / max(market_volume, 1)
        impact_slip = self.impact_coefficient * volume_ratio
        
        # Volatility component
        volatility_slip = self.volatility_factor * volatility
        
        # Direction matters - buying has positive slippage, selling negative
        direction = 1 if order.side == OrderSide.BUY else -1
        
        total_slippage = direction * (fixed_slip + impact_slip + volatility_slip)
        
        return total_slippage
    
    def apply_slippage(self, order: Order, market_price: float, 
                      **kwargs) -> float:
        """Apply slippage to get execution price"""
        slippage = self.calculate_slippage(order, market_price, **kwargs)
        execution_price = market_price * (1 + slippage)
        return max(0.01, execution_price)  # Ensure positive price


class CommissionModel:
    """Model for calculating trading commissions"""
    
    def __init__(self, commission_type: str = "fixed", 
                 rate: float = 0.0, min_commission: float = 0.0):
        """
        Initialize commission model
        
        Args:
            commission_type: "fixed", "per_share", "percentage"
            rate: Commission rate (depends on type)
            min_commission: Minimum commission per trade
        """
        self.commission_type = commission_type
        self.rate = rate
        self.min_commission = min_commission
    
    def calculate_commission(self, order: Order, execution_price: float) -> float:
        """Calculate commission for an order"""
        if self.commission_type == "fixed":
            commission = self.rate
        elif self.commission_type == "per_share":
            commission = self.rate * order.quantity
        elif self.commission_type == "percentage":
            trade_value = order.quantity * execution_price
            commission = trade_value * (self.rate / 100)
        else:
            raise ValueError(f"Unknown commission type: {self.commission_type}")
        
        return max(commission, self.min_commission)


class OrderExecutionEngine:
    """Advanced order execution with slippage and commissions"""
    
    def __init__(self, slippage_model: SlippageModel = None,
                 commission_model: CommissionModel = None):
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_model = commission_model or CommissionModel()
        self.execution_history: List[Dict] = []
    
    def execute_order(self, order: Order, market_tick: CandlestickTick) -> Dict:
        """
        Execute order with realistic market conditions
        
        Returns:
            Dictionary with execution details
        """
        # Determine base execution price
        if order.order_type == OrderType.MARKET:
            base_price = market_tick.close
        else:  # LIMIT
            base_price = order.price
        
        # Apply slippage for market orders
        if order.order_type == OrderType.MARKET:
            execution_price = self.slippage_model.apply_slippage(
                order, base_price, 
                market_volume=market_tick.volume,
                volatility=0.02  # Could calculate from recent price data
            )
        else:
            execution_price = base_price
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, execution_price)
        
        # Calculate total cost/proceeds
        trade_value = order.quantity * execution_price
        
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
        else:  # SELL
            total_cost = trade_value - commission
        
        execution_details = {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'order_price': order.price,
            'execution_price': execution_price,
            'commission': commission,
            'trade_value': trade_value,
            'total_cost': total_cost,
            'execution_time': market_tick.timestamp,
            'slippage': execution_price - base_price if order.order_type == OrderType.MARKET else 0
        }
        
        self.execution_history.append(execution_details)
        return execution_details
    
    def get_execution_statistics(self) -> Dict:
        """Get statistics on order executions"""
        if not self.execution_history:
            return {}
        
        total_trades = len(self.execution_history)
        total_commission = sum(e['commission'] for e in self.execution_history)
        total_slippage = sum(abs(e['slippage']) for e in self.execution_history)
        
        avg_commission = total_commission / total_trades
        avg_slippage = total_slippage / total_trades
        
        buy_trades = [e for e in self.execution_history if e['side'] == OrderSide.BUY]
        sell_trades = [e for e in self.execution_history if e['side'] == OrderSide.SELL]
        
        return {
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'avg_commission': avg_commission,
            'avg_slippage': avg_slippage,
            'avg_trade_size': sum(e['quantity'] for e in self.execution_history) / total_trades
        }


class OrderBookSimulator:
    """Simple order book simulation for realistic execution"""
    
    def __init__(self, spread_bps: float = 5.0):
        """
        Initialize order book simulator
        
        Args:
            spread_bps: Bid-ask spread in basis points
        """
        self.spread_bps = spread_bps
        self.bid_orders: List[Order] = []
        self.ask_orders: List[Order] = []
    
    def get_bid_ask(self, market_price: float) -> tuple:
        """Get current bid and ask prices"""
        spread = market_price * (self.spread_bps / 10000)
        bid = market_price - spread / 2
        ask = market_price + spread / 2
        return bid, ask
    
    def can_execute_limit_order(self, order: Order, market_tick: CandlestickTick) -> bool:
        """Check if a limit order can be executed"""
        if order.order_type != OrderType.LIMIT or not order.price:
            return False
        
        bid, ask = self.get_bid_ask(market_tick.close)
        
        if order.side == OrderSide.BUY:
            # Buy order executes if limit price >= ask or if price touched during the candle
            return order.price >= ask or order.price >= market_tick.low
        else:  # SELL
            # Sell order executes if limit price <= bid or if price touched during the candle
            return order.price <= bid or order.price <= market_tick.high
    
    def get_execution_price(self, order: Order, market_tick: CandlestickTick) -> float:
        """Get realistic execution price for order"""
        if order.order_type == OrderType.MARKET:
            bid, ask = self.get_bid_ask(market_tick.close)
            return ask if order.side == OrderSide.BUY else bid
        else:  # LIMIT
            return order.price


class AdvancedOrderTypes:
    """Support for advanced order types"""
    
    @staticmethod
    def create_stop_loss_order(symbol: str, quantity: int, stop_price: float,
                             side: OrderSide) -> Order:
        """Create a stop-loss order (simplified as limit order)"""
        # In a real implementation, this would be a separate order type
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=stop_price
        )
    
    @staticmethod
    def create_take_profit_order(symbol: str, quantity: int, 
                               target_price: float, side: OrderSide) -> Order:
        """Create a take-profit order"""
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=target_price
        )
    
    @staticmethod
    def create_bracket_orders(symbol: str, quantity: int, entry_price: float,
                            stop_loss: float, take_profit: float) -> List[Order]:
        """Create a bracket order (entry + stop loss + take profit)"""
        orders = []
        
        # Entry order
        entry_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=entry_price
        )
        orders.append(entry_order)
        
        # Stop loss (sell)
        stop_order = AdvancedOrderTypes.create_stop_loss_order(
            symbol, quantity, stop_loss, OrderSide.SELL
        )
        orders.append(stop_order)
        
        # Take profit (sell)
        profit_order = AdvancedOrderTypes.create_take_profit_order(
            symbol, quantity, take_profit, OrderSide.SELL
        )
        orders.append(profit_order)
        
        return orders


class OrderPerformanceTracker:
    """Track order execution performance"""
    
    def __init__(self):
        self.executions: List[Dict] = []
        self.metrics: Dict = {}
    
    def record_execution(self, execution_details: Dict):
        """Record an order execution"""
        self.executions.append(execution_details)
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        if not self.executions:
            return
        
        # Calculate fill rates, average slippage, etc.
        total_executions = len(self.executions)
        
        # Slippage analysis
        slippages = [abs(e.get('slippage', 0)) for e in self.executions]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        # Commission analysis
        commissions = [e.get('commission', 0) for e in self.executions]
        total_commission = sum(commissions)
        
        # Timing analysis
        execution_times = [e.get('execution_time') for e in self.executions if e.get('execution_time')]
        
        self.metrics = {
            'total_executions': total_executions,
            'avg_slippage': avg_slippage,
            'total_commission': total_commission,
            'avg_commission': total_commission / total_executions if total_executions > 0 else 0,
            'execution_count_by_side': {
                'BUY': len([e for e in self.executions if e.get('side') == OrderSide.BUY]),
                'SELL': len([e for e in self.executions if e.get('side') == OrderSide.SELL])
            }
        }
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            'summary': self.metrics,
            'recent_executions': self.executions[-10:],  # Last 10 executions
            'total_records': len(self.executions)
        }