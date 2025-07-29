"""
Pre-built pattern-based trading strategies.
"""

import time
from typing import Callable, Dict, Any, List
from datetime import datetime, timedelta
from ..core.models import PatternMatch, CandlestickTick
from ..core.types import OrderSide, PatternType

# Configurable strategy system imports handled dynamically to avoid circular imports


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


class UniversalPositionManager:
    """Universal position tracking and auto-sell for all strategy types"""
    
    def __init__(self, engine):
        self.engine = engine
        self.active_positions: Dict[str, Dict] = {}
        
    def track_new_position(self, symbol: str, order_id: str, entry_price: float, quantity: int):
        """Track a new position for auto-sell monitoring"""
        self.active_positions[symbol] = {
            'order_id': order_id,
            'entry_price': entry_price,
            'quantity': quantity,
            'candles_seen': 0,
            'red_candles_seen': 0,
            'entry_time': time.time()
        }
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"📊 Now tracking position: {symbol} - {quantity} shares @ ${entry_price:.2f}")
    
    def on_candle_completed(self, candle: CandlestickTick):
        """Monitor all tracked positions for sell conditions"""
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
        logger.info("🚨 UNIVERSAL POSITION UPDATE:")
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
    
    def _execute_sell(self, symbol: str, reason: str, current_price: float):
        """Execute sell order and remove position tracking"""
        if symbol not in self.active_positions:
            return
            
        position = self.active_positions[symbol]
        
        try:
            from ..core.types import OrderSide
            # Place sell order
            order_id = self.engine.place_market_order(symbol, OrderSide.SELL, position['quantity'])
            
            # Calculate P&L
            entry_price = position['entry_price']
            pnl = (current_price - entry_price) * position['quantity']
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info("")
            logger.info("💰 UNIVERSAL SELL EXECUTED:")
            logger.info(f"   📉 {symbol} position CLOSED | Reason: {reason}")
            logger.info(f"   🔄 Entry: ${entry_price:.2f} → Exit: ${current_price:.2f} | {position['quantity']} shares")
            logger.info(f"   💸 P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Order ID: {order_id}")
            logger.info("")
            
            # Remove position from tracking
            del self.active_positions[symbol]
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Failed to execute universal sell for {symbol}: {e}")


class BearishReversalSellStrategy:
    """
    Enhanced bearish reversal strategy with configurable sell conditions
    """
    
    def __init__(self, engine, confidence_threshold: float = 0.65, 
                 quantity: int = 100, stop_loss_pct: float = 0.0,
                 sell_after_candles: int = 2, sell_on_red_candle: bool = True):
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        self.quantity = quantity
        self.stop_loss_pct = stop_loss_pct  # Disabled (0.0)
        
        # Configurable sell conditions
        self.sell_after_candles = sell_after_candles  # Number of candles to wait before selling
        self.sell_on_red_candle = sell_on_red_candle  # Whether to sell immediately on red candle
        
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
        
        # Configurable sell condition 1: After N candles
        if position['candles_seen'] >= self.sell_after_candles:
            self._execute_sell(candle.symbol, f"{self.sell_after_candles} candles elapsed", candle.close)
            return
            
        # Configurable sell condition 2: After 1 red candle (if enabled)
        if self.sell_on_red_candle and position['red_candles_seen'] >= 1:
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
        
        # Create universal position manager
        self._universal_position_manager = UniversalPositionManager(engine)
        
        # Wrap engine's place_market_order to auto-track BUY orders
        self._original_place_market_order = engine.place_market_order
        engine.place_market_order = self._tracked_place_market_order
    
    def _tracked_place_market_order(self, symbol: str, side: OrderSide, quantity: int) -> str:
        """Wrapped place_market_order - Universal tracking disabled to avoid conflicts"""
        order_id = self._original_place_market_order(symbol, side, quantity)
        
        # Universal position tracking disabled - using strategy-specific tracking instead
        # The BearishReversalSellStrategy handles its own position tracking
        
        return order_id
    
    def add_strategy(self, name: str, strategy_func: Callable, pattern_type: PatternType = None):
        """Add a strategy to the factory"""
        self.active_strategies[name] = strategy_func
        
        if pattern_type:
            self.engine.register_pattern_strategy(pattern_type, strategy_func)
        else:
            # Assume it's an algorithm callback
            self.engine.register_algorithm(strategy_func)
    
    def create_conservative_setup(self):
        """Create a conservative trading setup - BULLISH REVERSAL ONLY"""
        # Disabled all other strategies - only bullish reversals
        # flat_top = PatternStrategies.create_flat_top_strategy(...)
        # self.add_strategy("conservative_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        # High confidence bullish reversals ONLY
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.75,
            quantity=100  # Increased quantity since it's the only strategy
        )
        self.add_strategy("conservative_reversal", reversal, PatternType.BULLISH_REVERSAL)
    
    def create_aggressive_setup(self):
        """Create an aggressive trading setup - BULLISH REVERSAL ONLY"""
        # Disabled all other strategies - only bullish reversals
        # flat_top = PatternStrategies.create_flat_top_strategy(...)
        # self.add_strategy("aggressive_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.6,  # Slightly higher confidence
            quantity=100
        )
        self.add_strategy("aggressive_reversal", reversal, PatternType.BULLISH_REVERSAL)
        
        # Disabled momentum strategy
        # momentum = PatternStrategies.create_momentum_strategy(self.engine)
        # self.add_strategy("momentum", momentum)
    
    def create_balanced_setup(self):
        """Create a balanced trading setup - BULLISH REVERSAL ONLY"""
        # Disabled all other strategies - only bullish reversals
        # flat_top = PatternStrategies.create_flat_top_strategy(...)
        # self.add_strategy("balanced_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        # Use ONLY the enhanced bearish reversal strategy with sell logic (no other strategies)
        # This strategy handles both buying on BULLISH_REVERSAL and selling with smart conditions
        # Configure to sell after 3 candles for more price movement opportunity
        bearish_strategy = self.create_bearish_reversal_with_sells(
            sell_after_candles=3,  # Sell after 3 candles for more price movement
            sell_on_red_candle=True  # Still sell on red candle too
        )
        # Note: Don't use add_strategy for this as it's handled differently
    
    def create_configurable_setup(self, config):
        """Create a strategy setup using the new configurable system"""
        try:
            # Dynamic import to avoid circular dependency
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from strategy_config import create_strategy_from_config
            
            # Create the configurable strategy
            configurable_strategy = create_strategy_from_config(config, self.engine)
            
            # Create pattern callback wrapper
            def pattern_callback(pattern):
                configurable_strategy.on_pattern_detected(pattern)
            
            # Register the pattern callback
            self.engine.register_pattern_strategy(config.buy_pattern_type, pattern_callback)
            
            # Store reference for candle callbacks
            self._configurable_strategy = configurable_strategy
            
            # Disable universal position manager to avoid conflicts
            # (ConfigurableStrategy handles its own position management)
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"✅ Created configurable strategy: {config.name}")
            
            return configurable_strategy
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Failed to create configurable strategy: {e}")
            raise
    
    def create_preset_setup(self, preset_name: str):
        """Create a strategy using a predefined preset configuration"""
        # Dynamic import to avoid circular dependency
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from strategy_config import get_preset_config
        
        config = get_preset_config(preset_name)
        return self.create_configurable_setup(config)
    
    def create_bearish_reversal_with_sells(self, sell_after_candles: int = 2, sell_on_red_candle: bool = True):
        """Create bearish reversal strategy with configurable sell conditions"""
        bearish_strategy = BearishReversalSellStrategy(
            self.engine,
            confidence_threshold=0.65,
            quantity=100,
            stop_loss_pct=0.0,  # Disabled
            sell_after_candles=sell_after_candles,
            sell_on_red_candle=sell_on_red_candle
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
        """Register strategy callbacks with RealTimeTradingEngine"""
        if hasattr(self, '_bearish_strategy'):
            # Register candle monitoring for bearish strategy
            rt_engine.register_candle_callback(self._bearish_strategy.on_candle_completed)
            
            # Register tick monitoring for stop loss
            def tick_monitor(tick_data):
                self._bearish_strategy.on_tick_received(tick_data.ticker, tick_data.price)
            
            rt_engine.register_tick_callback(tick_monitor)
        
        # Create universal position manager for all strategies
        if not hasattr(self, '_universal_position_manager'):
            self._universal_position_manager = UniversalPositionManager(self.engine)
        
        # Register candle monitoring for universal position tracking
        rt_engine.register_candle_callback(self._universal_position_manager.on_candle_completed)
    
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