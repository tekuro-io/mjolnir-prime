"""
Pre-built pattern-based trading strategies.
"""

from typing import Callable, Dict, Any, List
from datetime import datetime, timedelta
from ..core.models import PatternMatch, CandlestickTick
from ..core.types import OrderSide, PatternType


class PatternStrategies:
    """Pre-built pattern-based trading strategies"""
    
    @staticmethod
    def create_flat_top_strategy(engine, confidence_threshold: float = 0.7, 
                               quantity: int = 100, max_risk_per_trade: float = 0.02) -> Callable:
        """
        Create a flat top breakout strategy
        
        Args:
            engine: Trading engine instance
            confidence_threshold: Minimum confidence to trigger trade
            quantity: Number of shares to trade
            max_risk_per_trade: Maximum risk as percentage of portfolio
        """
        def strategy(pattern: PatternMatch):
            if pattern.confidence < confidence_threshold:
                return
            
            # Risk management - don't risk more than max_risk_per_trade of portfolio
            portfolio_value = engine.portfolio.get_portfolio_value(engine.current_prices)
            max_position_value = portfolio_value * max_risk_per_trade
            current_price = engine.current_prices.get(pattern.symbol, pattern.trigger_price)
            
            # Adjust quantity based on risk
            adjusted_quantity = min(quantity, int(max_position_value / current_price))
            
            if adjusted_quantity > 0:
                print(f"🚀 Flat top breakout: {pattern.symbol}")
                print(f"   Confidence: {pattern.confidence:.2f}")
                print(f"   Trigger price: ${pattern.trigger_price:.2f}")
                print(f"   Resistance level: ${pattern.metadata.get('resistance_level', 'N/A'):.2f}")
                print(f"   Trading {adjusted_quantity} shares")
                
                order_id = engine.place_market_order(pattern.symbol, OrderSide.BUY, adjusted_quantity)
                print(f"   Order placed: {order_id}")
        
        return strategy
    
    @staticmethod
    def create_reversal_strategy(engine, confidence_threshold: float = 0.6,
                               quantity: int = 50, use_recovery_strength: bool = True) -> Callable:
        """
        Create a bullish reversal strategy
        
        Args:
            engine: Trading engine instance
            confidence_threshold: Minimum confidence to trigger trade
            quantity: Number of shares to trade
            use_recovery_strength: Whether to adjust quantity based on recovery strength
        """
        def strategy(pattern: PatternMatch):
            if pattern.confidence < confidence_threshold:
                return
            
            recovery_strength = pattern.metadata.get('recovery_strength', 0.5)
            
            # Adjust quantity based on recovery strength if enabled
            if use_recovery_strength:
                adjusted_quantity = int(quantity * max(0.5, recovery_strength))
            else:
                adjusted_quantity = quantity
            
            if adjusted_quantity > 0:
                print(f"📈 Bullish reversal: {pattern.symbol}")
                print(f"   Confidence: {pattern.confidence:.2f}")
                print(f"   Recovery strength: {recovery_strength:.2f}")
                print(f"   Trading {adjusted_quantity} shares")
                
                order_id = engine.place_market_order(pattern.symbol, OrderSide.BUY, adjusted_quantity)
                print(f"   Order placed: {order_id}")
        
        return strategy
    
    @staticmethod
    def create_mean_reversion_strategy(engine, confidence_threshold: float = 0.65,
                                     quantity: int = 75) -> Callable:
        """
        Create a mean reversion strategy for bearish patterns
        """
        def strategy(pattern: PatternMatch):
            if (pattern.pattern_type == PatternType.BEARISH_REVERSAL and 
                pattern.confidence >= confidence_threshold):
                
                print(f"📉 Mean reversion opportunity: {pattern.symbol}")
                print(f"   Confidence: {pattern.confidence:.2f}")
                print(f"   Shorting {quantity} shares")
                
                # In a real system, you'd implement short selling
                # For now, we'll just log the opportunity
                print(f"   Short sell signal logged (not implemented)")
        
        return strategy
    
    @staticmethod
    def create_momentum_strategy(engine, lookback_periods: int = 5,
                               momentum_threshold: float = 0.02) -> Callable:
        """
        Create a momentum-based strategy that combines with patterns
        """
        def algorithm_callback(tick, new_patterns):
            # Calculate simple momentum
            if len(engine.pattern_detector.candle_history.get(tick.symbol, [])) >= lookback_periods:
                candles = list(engine.pattern_detector.candle_history[tick.symbol])
                recent_candles = candles[-lookback_periods:]
                
                price_change = (recent_candles[-1].close - recent_candles[0].close) / recent_candles[0].close
                
                # Strong momentum + pattern confirmation
                for pattern in new_patterns:
                    if (price_change > momentum_threshold and 
                        pattern.pattern_type in [PatternType.FLAT_TOP_BREAKOUT, PatternType.BULLISH_REVERSAL]):
                        
                        print(f"🚀 Momentum + Pattern confirmation: {tick.symbol}")
                        print(f"   Momentum: {price_change:.2%}")
                        print(f"   Pattern: {pattern.pattern_type.value}")
                        
                        order_id = engine.place_market_order(tick.symbol, OrderSide.BUY, 100)
                        print(f"   Momentum order: {order_id}")
        
        return algorithm_callback


class BearishReversalSellStrategy:
    """
    Enhanced bearish reversal strategy with smart sell conditions:
    - Sell after 2 candles
    - Sell after 1 red candle  
    - Sell on 0.05% stop loss
    """
    
    def __init__(self, engine, confidence_threshold: float = 0.65, 
                 quantity: int = 100, stop_loss_pct: float = 0.0):
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        self.quantity = quantity
        self.stop_loss_pct = stop_loss_pct  # Disabled (0.0)
        
        # Track active bearish reversal positions
        self.active_positions: Dict[str, Dict] = {}
        
    def on_pattern_detected(self, pattern: PatternMatch):
        """Handle bearish reversal pattern detection"""
        if (pattern.pattern_type == PatternType.BULLISH_REVERSAL and 
            pattern.confidence >= self.confidence_threshold):
            
            # Check if we already have a position in this symbol
            if pattern.symbol in self.active_positions:
                print(f"📉 Bearish reversal detected for {pattern.symbol} but position already exists")
                return
                
            # Enhanced logging for pattern detection
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info("")
            logger.info("🎯 PATTERN ALERT:")
            logger.info(f"   📈 BULLISH REVERSAL detected: {pattern.symbol}")
            logger.info(f"   💰 Entry: ${pattern.trigger_price:.2f} | Confidence: {pattern.confidence:.0%} | Stop Loss: DISABLED")
            
            try:
                # Place BUY order (we're buying for a potential reversal up)
                order_id = self.engine.place_market_order(pattern.symbol, OrderSide.BUY, self.quantity)
                
                # Track this position
                self.active_positions[pattern.symbol] = {
                    'order_id': order_id,
                    'entry_price': pattern.trigger_price,
                    'entry_time': pattern.timestamp,
                    'quantity': self.quantity,
                    'candles_seen': 0,
                    'red_candles_seen': 0,
                    'stop_loss_price': 0.0  # Disabled
                }
                
                logger.info(f"   ✅ BUY order placed: {self.quantity} shares | Order ID: {order_id}")
                logger.info("")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to place order: {e}")
                logger.info("")
    
    def on_candle_completed(self, candle: CandlestickTick):
        """Monitor candles for sell conditions"""
        if candle.symbol not in self.active_positions:
            return
            
        position = self.active_positions[candle.symbol]
        position['candles_seen'] += 1
        
        # Check if it's a red candle (close < open)
        is_red_candle = candle.close < candle.open
        if is_red_candle:
            position['red_candles_seen'] += 1
        
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("")
        logger.info("🚨 POSITION UPDATE:")
        logger.info(f"   📊 Monitoring {candle.symbol}: Candle {position['candles_seen']}, Red candles: {position['red_candles_seen']}, Price: ${candle.close:.2f}")
        logger.info("")
        
        # Sell condition 1: After 2 candles
        if position['candles_seen'] >= 2:
            self._execute_sell(candle.symbol, "2 candles elapsed", candle.close)
            return
            
        # Sell condition 2: After 1 red candle
        if position['red_candles_seen'] >= 1:
            self._execute_sell(candle.symbol, "red candle detected", candle.close)
            return
    
    def on_tick_received(self, symbol: str, price: float):
        """Monitor ticks for stop loss (currently disabled)"""
        # Stop loss functionality disabled
        pass
    
    def _execute_sell(self, symbol: str, reason: str, current_price: float):
        """Execute sell order and remove position tracking"""
        if symbol not in self.active_positions:
            return
            
        position = self.active_positions[symbol]
        
        try:
            # Place sell order
            order_id = self.engine.place_market_order(symbol, OrderSide.SELL, position['quantity'])
            
            # Calculate P&L
            entry_price = position['entry_price']
            pnl = (current_price - entry_price) * position['quantity']
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info("")
            logger.info("💰 SELL EXECUTED:")
            logger.info(f"   📉 {symbol} position CLOSED | Reason: {reason}")
            logger.info(f"   🔄 Entry: ${entry_price:.2f} → Exit: ${current_price:.2f} | {position['quantity']} shares")
            logger.info(f"   💸 P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Order ID: {order_id}")
            logger.info("")
            
            # Remove position from tracking
            del self.active_positions[symbol]
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Failed to execute sell for {symbol}: {e}")
    
    def get_active_positions(self) -> Dict:
        """Get current active positions"""
        return self.active_positions.copy()
    
    def clear_position(self, symbol: str):
        """Manually clear a position (for testing/admin)"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            print(f"🧹 Cleared position for {symbol}")


class StrategyFactory:
    """Factory class for creating and managing strategies"""
    
    def __init__(self, engine):
        self.engine = engine
        self.active_strategies: Dict[str, Callable] = {}
    
    def add_strategy(self, name: str, strategy_func: Callable, pattern_type: PatternType = None):
        """Add a strategy to the factory"""
        self.active_strategies[name] = strategy_func
        
        if pattern_type:
            self.engine.register_pattern_strategy(pattern_type, strategy_func)
        else:
            # Assume it's an algorithm callback
            self.engine.register_algorithm(strategy_func)
    
    def create_conservative_setup(self):
        """Create a conservative trading setup"""
        # High confidence flat top breakouts
        flat_top = PatternStrategies.create_flat_top_strategy(
            self.engine, 
            confidence_threshold=0.8, 
            quantity=50,
            max_risk_per_trade=0.01
        )
        self.add_strategy("conservative_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        # High confidence reversals
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.75,
            quantity=25
        )
        self.add_strategy("conservative_reversal", reversal, PatternType.BULLISH_REVERSAL)
    
    def create_aggressive_setup(self):
        """Create an aggressive trading setup"""
        # Lower confidence thresholds, higher quantities
        flat_top = PatternStrategies.create_flat_top_strategy(
            self.engine,
            confidence_threshold=0.6,
            quantity=200,
            max_risk_per_trade=0.05
        )
        self.add_strategy("aggressive_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.5,
            quantity=100
        )
        self.add_strategy("aggressive_reversal", reversal, PatternType.BULLISH_REVERSAL)
        
        # Add momentum strategy
        momentum = PatternStrategies.create_momentum_strategy(self.engine)
        self.add_strategy("momentum", momentum)
    
    def create_balanced_setup(self):
        """Create a balanced trading setup"""
        flat_top = PatternStrategies.create_flat_top_strategy(
            self.engine,
            confidence_threshold=0.7,
            quantity=100,
            max_risk_per_trade=0.025
        )
        self.add_strategy("balanced_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.65,
            quantity=75
        )
        self.add_strategy("balanced_reversal", reversal, PatternType.BULLISH_REVERSAL)
        
        # Add enhanced bearish reversal strategy with sell logic
        bearish_strategy = self.create_bearish_reversal_with_sells()
        # Note: Don't use add_strategy for this as it's handled differently
    
    def create_bearish_reversal_with_sells(self):
        """Create bearish reversal strategy with advanced sell conditions"""
        bearish_strategy = BearishReversalSellStrategy(
            self.engine,
            confidence_threshold=0.65,
            quantity=100,
            stop_loss_pct=0.0  # Disabled
        )
        
        # Create a pattern strategy function that wraps the bearish strategy
        def pattern_strategy_wrapper(pattern):
            bearish_strategy.on_pattern_detected(pattern)
        
        # Register with the engine for bullish reversal patterns
        self.engine.register_pattern_strategy(PatternType.BULLISH_REVERSAL, pattern_strategy_wrapper)
        
        # Store reference to the strategy for position tracking and callback registration
        self._bearish_strategy = bearish_strategy
        
        return bearish_strategy
    
    def register_with_realtime_engine(self, rt_engine):
        """Register bearish strategy callbacks with RealTimeTradingEngine"""
        if hasattr(self, '_bearish_strategy'):
            # Register candle monitoring
            rt_engine.register_candle_callback(self._bearish_strategy.on_candle_completed)
            
            # Register tick monitoring for stop loss
            def tick_monitor(tick_data):
                self._bearish_strategy.on_tick_received(tick_data.ticker, tick_data.price)
            
            rt_engine.register_tick_callback(tick_monitor)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about active strategies"""
        info = {
            'active_strategies': list(self.active_strategies.keys()),
            'strategy_count': len(self.active_strategies)
        }
        
        # Add bearish reversal position info if available
        for name, strategy in self.active_strategies.items():
            if isinstance(strategy, BearishReversalSellStrategy):
                info[f'{name}_positions'] = strategy.get_active_positions()
        
        return info